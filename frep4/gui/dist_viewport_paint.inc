// gui/dist_viewport_paint.inc — DistCanvas::paintEvent, split out so it sees the
// full DistViewport definition (it reads the owner's composited image).
namespace frep::gui {

void DistCanvas::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.fillRect(rect(), Qt::black);
    QImage snapshot;
    {
        std::lock_guard<std::mutex> lk(owner_->image_mu_);
        snapshot = owner_->image_;
    }
    if (snapshot.isNull()) return;
    // Fit the frame into the widget preserving aspect ratio.
    QSize target = snapshot.size().scaled(size(), Qt::KeepAspectRatio);
    QRect dst(QPoint(0, 0), target);
    dst.moveCenter(rect().center());
    p.drawImage(dst, snapshot);
}

}  // namespace frep::gui
