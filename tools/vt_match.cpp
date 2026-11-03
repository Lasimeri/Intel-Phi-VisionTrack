// vt_match.cpp: automatic correspondences between two long exposures, on the desktop host, offline: SIFT, the ratio test both ways, RANSAC on the fundamental matrix. Writes points for vt_calib.
//   vt_match A.pgm B.pgm OUT.txt [--ratio R] [--ransac PX] [--clahe C] [--mask-a X Y W H ...]
// A PGM of 8 or 16 bits is stretched to 8 and locally equalized, as what these frames show is dim. OpenCV is used here only: nothing of it reaches the card.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry/3d.hpp>

static cv::Mat load(const char *path, double clip)
{
    cv::Mat m = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (m.empty()) {
        std::fprintf(stderr, "%s: unreadable\n", path);
        std::exit(2);
    }
    cv::Mat f;
    m.convertTo(f, CV_32F);
    // A square-root curve before equalizing, the shadows being where the room is.
    double lo, hi;
    cv::minMaxLoc(f, &lo, &hi);
    f = (f - lo) / (hi - lo + 1e-9);
    cv::sqrt(f, f);
    cv::Mat g;
    f.convertTo(g, CV_8U, 255.0);
    cv::Ptr<cv::CLAHE> c = cv::createCLAHE(clip, cv::Size(16, 16));
    cv::Mat o;
    c->apply(g, o);
    return o;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::fprintf(stderr, "usage: vt_match A.pgm B.pgm OUT.txt [--ratio R] [--ransac PX] [--clahe C]\n");
        return 2;
    }
    double ratio = 0.8, ransac = 2.0, clip = 3.0;
    cv::Mat Hb;
    for (int i = 4; i + 1 < argc; i++) {
        if (!std::strcmp(argv[i], "--ratio")) ratio = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--ransac")) ransac = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--clahe")) clip = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--hb") && i + 9 < argc) {
            // A homography for image B, into A's frame: a rotation removed, so only displacement is left between the pictures. Matches are mapped back to B's own pixels.
            Hb = cv::Mat(3, 3, CV_64F);
            for (int k = 0; k < 9; k++)
                Hb.at<double>(k / 3, k % 3) = std::atof(argv[++i]);
        }
    }
    cv::setRNGSeed(12345);
    cv::Mat a = load(argv[1], clip), b = load(argv[2], clip);
    if (!Hb.empty()) {
        cv::Mat warped;
        cv::warpPerspective(b, warped, Hb, a.size(), cv::INTER_LINEAR);
        b = warped;
    }
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(0, 4, 0.02, 12.0, 1.6);
    std::vector<cv::KeyPoint> ka, kb;
    cv::Mat da, db;
    sift->detectAndCompute(a, cv::noArray(), ka, da);
    sift->detectAndCompute(b, cv::noArray(), kb, db);
    cv::BFMatcher m(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> ab, ba;
    m.knnMatch(da, db, ab, 2);
    m.knnMatch(db, da, ba, 2);
    std::vector<cv::Point2f> pa, pb;
    for (const auto &v : ab) {
        if (v.size() < 2 || v[0].distance > ratio * v[1].distance)
            continue;
        const auto &w = ba[v[0].trainIdx];
        if (w.size() < 2 || w[0].distance > ratio * w[1].distance || w[0].trainIdx != v[0].queryIdx)
            continue;
        pa.push_back(ka[v[0].queryIdx].pt);
        pb.push_back(kb[v[0].trainIdx].pt);
    }
    std::printf("%s: %zu keypoints, %s: %zu keypoints; mutual ratio-test matches %zu", argv[1], ka.size(), argv[2], kb.size(), pa.size());
    if (pa.size() < 8) {
        std::printf("; too few\n");
        return 1;
    }
    std::vector<uchar> inl;
    cv::Mat F = cv::findFundamentalMat(pa, pb, cv::USAC_MAGSAC, ransac, 0.9999, 200000, inl);
    if (!Hb.empty()) {
        std::vector<cv::Point2f> back;
        cv::perspectiveTransform(pb, back, Hb.inv());
        pb = back;
    }
    int n = 0;
    FILE *f = std::fopen(argv[3], "w");
    if (!f)
        return 2;
    std::fprintf(f, "# %s, %s: SIFT, ratio %.2f, MAGSAC %.1f px\n", argv[1], argv[2], ratio, ransac);
    for (size_t i = 0; i < pa.size(); i++)
        if (i < inl.size() && inl[i]) {
            std::fprintf(f, "%.3f %.3f %.3f %.3f sift%d\n", pa[i].x, pa[i].y, pb[i].x, pb[i].y, n);
            n++;
        }
    std::fclose(f);
    std::printf("; inliers %d\n", n);
    return 0;
}
