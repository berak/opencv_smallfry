/*
The MIT License(MIT)

Copyright(c) 2015 Yang Cao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "face_x.h"

#include <algorithm>
#include <stdexcept>
#include <cassert>

using namespace std;



//
// Utils ===============================================
//

void Transform::Apply(vector<cv::Point2d> &x, bool need_translation) const
{
    for (size_t i=0; i<x.size(); i++)
    {
        cv::Point2d &p = x[i];
        cv::Matx21d v;
        v(0) = p.x;
        v(1) = p.y;
        v = scale_rotation * v;
        if (need_translation)
            v += translation;
        p.x = v(0);
        p.y = v(1);
    }
}

Transform Procrustes(const vector<cv::Point2d> &x, const vector<cv::Point2d> &y)
{
    assert(x.size() == y.size());
    int landmark_count = x.size();
    double X1 = 0, X2 = 0, Y1 = 0, Y2 = 0, Z = 0, W = landmark_count;
    double C1 = 0, C2 = 0;

    for (int i = 0; i < landmark_count; ++i)
    {
        X1 += x[i].x;
        X2 += y[i].x;
        Y1 += x[i].y;
        Y2 += y[i].y;
        Z += Sqr(y[i].x) + Sqr(y[i].y);
        C1 += x[i].x * y[i].x + x[i].y * y[i].y;
        C2 += x[i].y * y[i].x - x[i].x * y[i].y;
    }

    cv::Matx44d A(X2, -Y2, W, 0,
        Y2, X2, 0, W,
        Z, 0, X2, Y2,
        0, Z, -Y2, X2);
    cv::Matx41d b(X1, Y1, C1, C2);
    cv::Matx41d solution = A.inv() * b;

    Transform result;
    result.scale_rotation(0, 0) = solution(0);
    result.scale_rotation(0, 1) = -solution(1);
    result.scale_rotation(1, 0) = solution(1);
    result.scale_rotation(1, 1) = solution(0);
    result.translation(0) = solution(2);
    result.translation(1) = solution(3);
    return result;
}

vector<cv::Point2d> ShapeAdjustment(const vector<cv::Point2d> &shape,
    const vector<cv::Point2d> &offset)
{
    assert(shape.size() == offset.size());
    vector<cv::Point2d> result(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
        result[i] = shape[i] + offset[i];
    return result;
}

vector<cv::Point2d> MapShape(cv::Rect original_face_rect,
    const vector<cv::Point2d> original_landmarks, cv::Rect new_face_rect)
{
    vector<cv::Point2d> result;
    for (size_t i=0; i<original_landmarks.size(); ++i)
    {
        const cv::Point2d &landmark = original_landmarks[i];
        result.push_back(landmark);
        result.back() -= cv::Point2d(original_face_rect.x, original_face_rect.y);
        result.back().x *=
            static_cast<double>(new_face_rect.width) / original_face_rect.width;
        result.back().y *=
            static_cast<double>(new_face_rect.height) / original_face_rect.height;
        result.back() += cv::Point2d(new_face_rect.x, new_face_rect.y);
    }
    return result;
}


//
// Fern ===============================================
//

void Fern::ApplyMini(cv::Mat features, std::vector<double> &coeffs)const
{
    int outputs_index = 0;
    for (size_t i = 0; i < features_index.size(); ++i)
    {
        pair<int, int> feature = features_index[i];
        double p1 = features.at<double>(feature.first);
        double p2 = features.at<double>(feature.second);
        outputs_index |= (p1 - p2 > thresholds[i]) << i;
    }

    const vector<pair<int, double>> &output = outputs_mini[outputs_index];
    for (size_t i = 0; i < output.size(); ++i)
        coeffs[output[i].first] += output[i].second;
}

void Fern::read(const cv::FileNode &fn)
{
    thresholds.clear();
    features_index.clear();
    outputs_mini.clear();
    fn["thresholds"] >> thresholds;
    cv::FileNode features_index_node = fn["features_index"];
    for (cv::FileNodeIterator it = features_index_node.begin(); it != features_index_node.end(); ++it)
    {
        pair<int, int> feature_index;
        (*it)["first"] >> feature_index.first;
        (*it)["second"] >> feature_index.second;
        features_index.push_back(feature_index);
    }
    cv::FileNode outputs_mini_node = fn["outputs_mini"];
    for (cv::FileNodeIterator it = outputs_mini_node.begin(); it != outputs_mini_node.end(); ++it)
    {
        vector<std::pair<int, double>> output;
        cv::FileNode output_node = *it;
        for (cv::FileNodeIterator it2 = output_node.begin(); it2 != output_node.end(); ++it2)
            output.push_back(make_pair((*it2)["index"], (*it2)["coeff"]));
        outputs_mini.push_back(output);
    }
}

void read(const cv::FileNode& node, Fern &f, const Fern&)
{
    if (node.empty())
        throw runtime_error("Model file is corrupt!");
    else
        f.read(node);
}





//
// Regressor ===============================================
//



vector<cv::Point2d> Regressor::Apply(const Transform &t,
    cv::Mat image, const std::vector<cv::Point2d> &init_shape) const
{
    cv::Mat pixels_val(1, pixels_.size(), CV_64FC1);
    vector<cv::Point2d> offsets(pixels_.size());
    for (size_t j = 0; j < pixels_.size(); ++j)
        offsets[j] = pixels_[j].second;
    t.Apply(offsets, false);

    double *p = pixels_val.ptr<double>(0);
    for (size_t j = 0; j < pixels_.size(); ++j)
    {
        cv::Point pixel_pos = init_shape[pixels_[j].first] + offsets[j];
        if (pixel_pos.inside(cv::Rect(0, 0, image.cols, image.rows)))
            p[j] = image.at<uchar>(pixel_pos);
        else
            p[j] = 0;
    }

    vector<double> coeffs(base_.cols);
    for (size_t i = 0; i < ferns_.size(); ++i)
        ferns_[i].ApplyMini(pixels_val, coeffs);

    cv::Mat result_mat = base_ * cv::Mat(coeffs);

    vector<cv::Point2d> result(init_shape.size());
    {
        for (size_t i = 0; i < result.size(); ++i)
        {
            result[i].x = result_mat.at<double>(i * 2);
            result[i].y = result_mat.at<double>(i * 2 + 1);
        }
    }
    return result;
}

void Regressor::read(const cv::FileNode &fn)
{
    pixels_.clear();
    ferns_.clear();
    cv::FileNode pixels_node = fn["pixels"];
    for (cv::FileNodeIterator it = pixels_node.begin(); it != pixels_node.end(); ++it)
    {
        pair<int, cv::Point2d> pixel;
        (*it)["first"] >> pixel.first;
        (*it)["second"] >> pixel.second;
        pixels_.push_back(pixel);
    }
    cv::FileNode ferns_node = fn["ferns"];
    for (cv::FileNodeIterator it = ferns_node.begin(); it != ferns_node.end(); ++it)
    {
        Fern f;
        *it >> f;
        ferns_.push_back(f);
    }
    fn["base"] >> base_;
}

void read(const cv::FileNode& node, Regressor& r, const Regressor&)
{
    if (node.empty())
        throw runtime_error("Model file is corrupt!");
    else
        r.read(node);
}


//
// FaceX ===============================================
//

FaceX::FaceX(const string & filename)
{
    cv::FileStorage model_file;
    model_file.open(filename, cv::FileStorage::READ);
    if (!model_file.isOpened())
        throw runtime_error("Cannot open model file \"" + filename + "\".");

    model_file["mean_shape"] >> mean_shape_;
    cv::FileNode fn = model_file["test_init_shapes"];
    for (cv::FileNodeIterator it = fn.begin(); it != fn.end(); ++it)
    {
        vector<cv::Point2d> shape;
        *it >> shape;
        test_init_shapes_.push_back(shape);
    }
    fn = model_file["stage_regressors"];
    for (cv::FileNodeIterator it = fn.begin(); it != fn.end(); ++it)
    {
        Regressor r;
        *it >> r;
        stage_regressors_.push_back(r);
    }
}

vector<cv::Point2d> FaceX::Alignment(cv::Mat image, cv::Rect face_rect) const
{
    vector<vector<double>> all_results(test_init_shapes_[0].size() * 2);
    for (size_t i = 0; i < test_init_shapes_.size(); ++i)
    {
        vector<cv::Point2d> init_shape = MapShape(cv::Rect(0, 0, 1, 1),
            test_init_shapes_[i], face_rect);
        for (size_t j = 0; j < stage_regressors_.size(); ++j)
        {
            Transform t = Procrustes(init_shape, mean_shape_);
            vector<cv::Point2d> offset =
                stage_regressors_[j].Apply(t, image, init_shape);
            t.Apply(offset, false);
            init_shape = ShapeAdjustment(init_shape, offset);
        }

        for (size_t i = 0; i < init_shape.size(); ++i)
        {
            all_results[i * 2].push_back(init_shape[i].x);
            all_results[i * 2 + 1].push_back(init_shape[i].y);
        }
    }

    vector<cv::Point2d> result(test_init_shapes_[0].size());
    for (size_t i = 0; i < result.size(); ++i)
    {
        nth_element(all_results[i * 2].begin(),
            all_results[i * 2].begin() + test_init_shapes_.size() / 2,
            all_results[i * 2].end());
        result[i].x = all_results[i * 2][test_init_shapes_.size() / 2];
        nth_element(all_results[i * 2 + 1].begin(),
            all_results[i * 2 + 1].begin() + test_init_shapes_.size() / 2,
            all_results[i * 2 + 1].end());
        result[i].y = all_results[i * 2 + 1][test_init_shapes_.size() / 2];
    }
    return result;
}

vector<cv::Point2d> FaceX::Alignment(cv::Mat image,
    vector<cv::Point2d> initial_landmarks) const
{
    vector<vector<double>> all_results(test_init_shapes_[0].size() * 2);
    for (size_t i = 0; i < test_init_shapes_.size(); ++i)
    {
        Transform t = Procrustes(initial_landmarks, test_init_shapes_[i]);
        vector<cv::Point2d> init_shape = test_init_shapes_[i];
        t.Apply(init_shape);
        for (size_t j = 0; j < stage_regressors_.size(); ++j)
        {
            Transform t = Procrustes(init_shape, mean_shape_);
            vector<cv::Point2d> offset =
                stage_regressors_[j].Apply(t, image, init_shape);
            t.Apply(offset, false);
            init_shape = ShapeAdjustment(init_shape, offset);
        }

        for (size_t i = 0; i < init_shape.size(); ++i)
        {
            all_results[i * 2].push_back(init_shape[i].x);
            all_results[i * 2 + 1].push_back(init_shape[i].y);
        }
    }

    vector<cv::Point2d> result(test_init_shapes_[0].size());
    for (size_t i = 0; i < result.size(); ++i)
    {
        nth_element(all_results[i * 2].begin(),
            all_results[i * 2].begin() + test_init_shapes_.size() / 2,
            all_results[i * 2].end());
        result[i].x = all_results[i * 2][test_init_shapes_.size() / 2];
        nth_element(all_results[i * 2 + 1].begin(),
            all_results[i * 2 + 1].begin() + test_init_shapes_.size() / 2,
            all_results[i * 2 + 1].end());
        result[i].y = all_results[i * 2 + 1][test_init_shapes_.size() / 2];
    }
    return result;
}
