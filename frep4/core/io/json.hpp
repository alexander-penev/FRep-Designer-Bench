#pragma once
// core/io/json.hpp
//
// Minimal JSON — only for scene serialization.
// No external dependencies (nlohmann would be overkill for a PoC).
// Supports: object, array, string, number, bool, null.

#include <cctype>
#include <cstdlib>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace frep::json {

class Value;
using Array  = std::vector<Value>;
using Object = std::map<std::string, Value>;

class Value {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Value() : type_(Type::Null) {}
    Value(bool b) : type_(Type::Bool), bool_(b) {}
    Value(double n) : type_(Type::Number), num_(n) {}
    Value(int n) : type_(Type::Number), num_(static_cast<double>(n)) {}
    Value(float n) : type_(Type::Number), num_(static_cast<double>(n)) {}
    Value(const char* s) : type_(Type::String), str_(s) {}
    Value(std::string s) : type_(Type::String), str_(std::move(s)) {}
    Value(Array a) : type_(Type::Array), arr_(std::move(a)) {}
    Value(Object o) : type_(Type::Object), obj_(std::move(o)) {}

    Type type() const { return type_; }
    bool is_object() const { return type_ == Type::Object; }
    bool is_array()  const { return type_ == Type::Array; }
    bool is_string() const { return type_ == Type::String; }
    bool is_number() const { return type_ == Type::Number; }

    bool        as_bool()   const { return bool_; }
    double      as_number() const { return num_; }
    float       as_float()  const { return static_cast<float>(num_); }
    int         as_int()    const { return static_cast<int>(num_); }
    const std::string& as_string() const { return str_; }
    const Array&  as_array()  const { return arr_; }
    const Object& as_object() const { return obj_; }
    Array&  as_array()  { return arr_; }
    Object& as_object() { return obj_; }

    // Object access
    const Value& operator[](const std::string& key) const {
        auto it = obj_.find(key);
        if (it == obj_.end())
            throw std::runtime_error("JSON: missing key '" + key + "'");
        return it->second;
    }
    bool has(const std::string& key) const {
        return type_ == Type::Object && obj_.count(key) > 0;
    }

    // ── Serialize ─────────────────────────────────────────────────────────────
    std::string dump(int indent = 2) const {
        std::ostringstream os;
        write(os, indent, 0);
        return os.str();
    }

    // ── Parse ─────────────────────────────────────────────────────────────────
    static Value parse(const std::string& text) {
        std::size_t pos = 0;
        skip_ws(text, pos);
        Value v = parse_value(text, pos);
        skip_ws(text, pos);
        if (pos != text.size())
            throw std::runtime_error("JSON: trailing characters after value");
        return v;
    }

private:
    Type        type_;
    bool        bool_ = false;
    double      num_  = 0.0;
    std::string str_;
    Array       arr_;
    Object      obj_;

    void write(std::ostringstream& os, int indent, int depth) const {
        std::string pad(static_cast<std::size_t>(indent * depth), ' ');
        std::string pad1(static_cast<std::size_t>(indent * (depth + 1)), ' ');
        switch (type_) {
            case Type::Null:   os << "null"; break;
            case Type::Bool:   os << (bool_ ? "true" : "false"); break;
            case Type::Number: {
                // whole numbers without .0; otherwise with a decimal point
                if (num_ == static_cast<long long>(num_))
                    os << static_cast<long long>(num_);
                else
                    os << num_;
                break;
            }
            case Type::String: write_string(os, str_); break;
            case Type::Array: {
                if (arr_.empty()) { os << "[]"; break; }
                os << "[\n";
                for (std::size_t i = 0; i < arr_.size(); ++i) {
                    os << pad1;
                    arr_[i].write(os, indent, depth + 1);
                    if (i + 1 < arr_.size()) os << ",";
                    os << "\n";
                }
                os << pad << "]";
                break;
            }
            case Type::Object: {
                if (obj_.empty()) { os << "{}"; break; }
                os << "{\n";
                std::size_t i = 0;
                for (const auto& [k, v] : obj_) {
                    os << pad1;
                    write_string(os, k);
                    os << ": ";
                    v.write(os, indent, depth + 1);
                    if (++i < obj_.size()) os << ",";
                    os << "\n";
                }
                os << pad << "}";
                break;
            }
        }
    }

    static void write_string(std::ostringstream& os, const std::string& s) {
        os << '"';
        for (char c : s) {
            switch (c) {
                case '"':  os << "\\\""; break;
                case '\\': os << "\\\\"; break;
                case '\n': os << "\\n";  break;
                case '\t': os << "\\t";  break;
                case '\r': os << "\\r";  break;
                default:   os << c;
            }
        }
        os << '"';
    }

    static void skip_ws(const std::string& t, std::size_t& p) {
        while (p < t.size() && std::isspace(static_cast<unsigned char>(t[p]))) ++p;
    }

    static Value parse_value(const std::string& t, std::size_t& p) {
        skip_ws(t, p);
        if (p >= t.size()) throw std::runtime_error("JSON: unexpected end");
        char c = t[p];
        if (c == '{') return parse_object(t, p);
        if (c == '[') return parse_array(t, p);
        if (c == '"') return Value(parse_string(t, p));
        if (c == 't' || c == 'f') return parse_bool(t, p);
        if (c == 'n') { p += 4; return Value(); }   // null
        return parse_number(t, p);
    }

    static Value parse_object(const std::string& t, std::size_t& p) {
        Object o;
        ++p;  // {
        skip_ws(t, p);
        if (p < t.size() && t[p] == '}') { ++p; return Value(std::move(o)); }
        while (true) {
            skip_ws(t, p);
            std::string key = parse_string(t, p);
            skip_ws(t, p);
            if (p >= t.size() || t[p] != ':')
                throw std::runtime_error("JSON: expected ':'");
            ++p;
            o[key] = parse_value(t, p);
            skip_ws(t, p);
            if (p >= t.size()) throw std::runtime_error("JSON: unterminated object");
            if (t[p] == ',') { ++p; continue; }
            if (t[p] == '}') { ++p; break; }
            throw std::runtime_error("JSON: expected ',' or '}'");
        }
        return Value(std::move(o));
    }

    static Value parse_array(const std::string& t, std::size_t& p) {
        Array a;
        ++p;  // [
        skip_ws(t, p);
        if (p < t.size() && t[p] == ']') { ++p; return Value(std::move(a)); }
        while (true) {
            a.push_back(parse_value(t, p));
            skip_ws(t, p);
            if (p >= t.size()) throw std::runtime_error("JSON: unterminated array");
            if (t[p] == ',') { ++p; continue; }
            if (t[p] == ']') { ++p; break; }
            throw std::runtime_error("JSON: expected ',' or ']'");
        }
        return Value(std::move(a));
    }

    static std::string parse_string(const std::string& t, std::size_t& p) {
        if (t[p] != '"') throw std::runtime_error("JSON: expected string");
        ++p;
        std::string out;
        while (p < t.size() && t[p] != '"') {
            char c = t[p++];
            if (c == '\\' && p < t.size()) {
                char e = t[p++];
                switch (e) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case '"': out += '"';  break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    default: out += e;
                }
            } else {
                out += c;
            }
        }
        if (p >= t.size()) throw std::runtime_error("JSON: unterminated string");
        ++p;  // "
        return out;
    }

    static Value parse_number(const std::string& t, std::size_t& p) {
        std::size_t start = p;
        if (p < t.size() && (t[p] == '-' || t[p] == '+')) ++p;
        while (p < t.size() &&
               (std::isdigit(static_cast<unsigned char>(t[p])) ||
                t[p] == '.' || t[p] == 'e' || t[p] == 'E' ||
                t[p] == '+' || t[p] == '-'))
            ++p;
        return Value(std::strtod(t.substr(start, p - start).c_str(), nullptr));
    }

    static Value parse_bool(const std::string& t, std::size_t& p) {
        if (t.compare(p, 4, "true") == 0)  { p += 4; return Value(true);  }
        if (t.compare(p, 5, "false") == 0) { p += 5; return Value(false); }
        throw std::runtime_error("JSON: invalid bool");
    }
};

} // namespace frep::json
