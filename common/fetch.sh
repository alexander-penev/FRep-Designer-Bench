# Shared system-resolution helpers. POSIX sh. Source from *_bench/build.sh.
#
#   resolve_root VAR default_dir   -> echoes the root; fetches if dir empty/missing
#   fetch_<name>                   -> per-system download recipe
#   apply_patches ROOT PATCHDIR    -> idempotent: skips already-applied hunks
#
set -e
: "${SUITE_DIR:?build.sh must export SUITE_DIR before sourcing fetch.sh}"

is_empty() { [ ! -d "$1" ] || [ -z "$(ls -A "$1" 2>/dev/null | grep -vx .gitkeep)" ]; }

fetch_git() { # dir url
    mkdir -p "$1"; rm -f "$1/.gitkeep"      # git clone refuses non-empty dirs
    git clone --depth 1 "$2" "$1"
}
fetch_frep4()    { fetch_git "$1" https://github.com/alexander-penev/FRep-Designer.git; }
fetch_libfive()  { fetch_git "$1" https://github.com/libfive/libfive; }
fetch_mpr()      { fetch_git "$1" https://github.com/mkeeter/mpr; }
fetch_hyperfun() {
    mkdir -p "$1"; rm -f "$1/.gitkeep"
    command -v 7z >/dev/null || { echo "hyperfun: p7zip (7z) required" >&2; exit 3; }
    curl -sL -o "$1/hyperfun_source.7z" \
        "https://downloads.sourceforge.net/project/hyperfun/hyperfun_source.7z"
    7z x -y -o"$1" "$1/hyperfun_source.7z" >/dev/null
    rm -f "$1/hyperfun_source.7z"
}

resolve_root() { # VAR name  (dir = $SUITE_DIR/name)
    eval v=\$$1
    d=${v:-$SUITE_DIR/$2}
    if is_empty "$d"; then
        echo "== $2: fetching into $d" >&2
        fetch_$2 "$d"
    fi
    echo "$d"
}

apply_patches() { # ROOT PATCHDIR
    [ -d "$2" ] || return 0
    for p in "$2"/*.patch; do
        [ -f "$p" ] || continue
        # already applied? (reverse-applies cleanly) -> skip
        if patch -p1 -R -s -f -t --dry-run -d "$1" < "$p" >/dev/null 2>&1; then
            echo "== patch $(basename "$p"): already applied"
        elif patch -p1 -N -s -f -t -r /dev/null -d "$1" < "$p"; then
            echo "== patch $(basename "$p"): applied"
        else
            echo "== patch $(basename "$p"): FAILED for $1" >&2
        fi
    done
    for a in "$2"/*.add; do            # NAME.add -> copied to ROOT/<embedded path>
        [ -f "$a" ] || continue
        rel=$(sed -n '1s|^// *dest: *||p' "$a")
        [ -n "$rel" ] || continue
        [ -f "$1/$rel" ] || { mkdir -p "$1/$(dirname "$rel")"; tail -n +2 "$a" > "$1/$rel"; echo "== added $rel"; }
    done
}
