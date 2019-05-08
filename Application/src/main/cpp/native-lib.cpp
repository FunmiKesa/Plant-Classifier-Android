#include <jni.h>
#include <string>
#include <algorithm>
#include <iostream>
#include <vector>
#include <numeric>      // std::iota
#include <math.h>
#define PROTOBUF_USE_DLLS 1
#define CAFFE2_USE_LITE_PROTO 1

#include <caffe2/predictor/predictor.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/timer.h>

#include "caffe2/core/init.h"
#include <caffe2/core/tensor.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <ATen/ATen.h>
#include "classes.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define IMG_H 224
#define IMG_W 224
#define IMG_C 3
#define MAX_DATA_SIZE IMG_H * IMG_W * IMG_C
#define alog(...) __android_log_print(ANDROID_LOG_ERROR, "teamfuture", __VA_ARGS__);

static caffe2::NetDef _initNet, _predictNet;
static caffe2::Predictor *_predictor;
static char raw_data[MAX_DATA_SIZE];
static float input_data[MAX_DATA_SIZE];
static caffe2::Workspace ws;

//template <typename T>
//vector<size_t> sort_indexes(const vector<T> &v) {
//
//    // initialize original index locations
//    vector<size_t> idx(v.size());
//    iota(idx.begin(), idx.end(), 0);
//
//    // sort indexes based on comparing values in v
//    sort(idx.begin(), idx.end(),
//         [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
//
//    return idx;
//}

// A function to load the NetDefs from protobufs.
void loadToNetDef(AAssetManager* mgr, caffe2::NetDef* net, const char *filename) {
    AAsset* asset = AAssetManager_open(mgr, filename, AASSET_MODE_BUFFER);
    assert(asset != nullptr);
    const void *data = AAsset_getBuffer(asset);
    assert(data != nullptr);
    off_t len = AAsset_getLength(asset);
    assert(len != 0);
    if (!net->ParseFromArray(data, len)) {
        alog("Couldn't parse net from data.\n");
    }
    AAsset_close(asset);
}

extern "C"
void
Java_teamfuture_plantclassifier_Camera2BasicFragment_initCaffe2(
        JNIEnv* env,
        jobject /* this */,
        jobject assetManager) {
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    alog("Attempting to load protobuf netdefs...");
//    loadToNetDef(mgr, &_initNet,   "resnet18_init_net_v1.pb");
//    loadToNetDef(mgr, &_predictNet,"resnet18_predict_net_v1.pb");
    loadToNetDef(mgr, &_initNet,   "init_net.pb");
    loadToNetDef(mgr, &_predictNet,"predict_net.pb");
    alog("done.");
    alog("Instantiating predictor...");
    _predictor = new caffe2::Predictor(_initNet, _predictNet);
    alog("done.")
}

float avg_fps = 0.0;
float total_fps = 0.0;
int iters_fps = 10;

extern "C"
JNIEXPORT jstring JNICALL
Java_teamfuture_plantclassifier_Camera2BasicFragment_classificationFromCaffe2(
        JNIEnv *env,
        jobject /* this */,
        jint h, jint w, jbyteArray Y, jbyteArray U, jbyteArray V,
        jint rowStride, jint pixelStride,
        jboolean infer_HWC) {
    if (!_predictor) {
        return env->NewStringUTF("Loading...");
    }
    jsize Y_len = env->GetArrayLength(Y);
    jbyte * Y_data = env->GetByteArrayElements(Y, 0);
    assert(Y_len <= MAX_DATA_SIZE);
    jsize U_len = env->GetArrayLength(U);
    jbyte * U_data = env->GetByteArrayElements(U, 0);
    assert(U_len <= MAX_DATA_SIZE);
    jsize V_len = env->GetArrayLength(V);
    jbyte * V_data = env->GetByteArrayElements(V, 0);
    assert(V_len <= MAX_DATA_SIZE);

#define min(a,b) ((a) > (b)) ? (b) : (a)
#define max(a,b) ((a) > (b)) ? (a) : (b)

    auto h_offset = max(0, (h - IMG_H) / 2);
    auto w_offset = max(0, (w - IMG_W) / 2);

    auto iter_h = IMG_H;
    auto iter_w = IMG_W;
    if (h < IMG_H) {
        iter_h = h;
    }
    if (w < IMG_W) {
        iter_w = w;
    }

    for (auto i = 0; i < iter_h; ++i) {
        jbyte* Y_row = &Y_data[(h_offset + i) * w];
        jbyte* U_row = &U_data[(h_offset + i) / 2 * rowStride];
        jbyte* V_row = &V_data[(h_offset + i) / 2 * rowStride];
        for (auto j = 0; j < iter_w; ++j) {
            // Tested on Pixel and S7.
            char y = Y_row[w_offset + j];
            char u = U_row[pixelStride * ((w_offset+j)/pixelStride)];
            char v = V_row[pixelStride * ((w_offset+j)/pixelStride)];

            float b_mean = 104.00698793f;
            float g_mean = 116.66876762f;
            float r_mean = 122.67891434f;

            auto b_i = 0 * IMG_H * IMG_W + j * IMG_W + i;
            auto g_i = 1 * IMG_H * IMG_W + j * IMG_W + i;
            auto r_i = 2 * IMG_H * IMG_W + j * IMG_W + i;

            if (infer_HWC) {
                b_i = (j * IMG_W + i) * IMG_C;
                g_i = (j * IMG_W + i) * IMG_C + 1;
                r_i = (j * IMG_W + i) * IMG_C + 2;
            }
            /*
              R = Y + 1.402 (V-128)
              G = Y - 0.34414 (U-128) - 0.71414 (V-128)
              B = Y + 1.772 (U-V)
             */
            input_data[r_i] = -r_mean + (float) ((float) min(255., max(0., (float) (y + 1.402 * (v - 128)))));
            input_data[g_i] = -g_mean + (float) ((float) min(255., max(0., (float) (y - 0.34414 * (u - 128) - 0.71414 * (v - 128)))));
            input_data[b_i] = -b_mean + (float) ((float) min(255., max(0., (float) (y + 1.772 * (u - v)))));

        }
    }

    caffe2::TensorCPU input;
    if (infer_HWC) {
        input = caffe2::Tensor(std::vector<int>({IMG_H, IMG_W, IMG_C}), caffe2::CPU);
    } else {
        input = caffe2::Tensor(std::vector<int>({1, IMG_C, IMG_H, IMG_W}), caffe2::CPU);
    }
    memcpy(input.mutable_data<float>(), input_data, IMG_H * IMG_W * IMG_C * sizeof(float));
    std::vector<caffe2::TensorCPU> input_vec({input});
    std::vector<caffe2::TensorCPU> output_vec(1);
    caffe2::Timer t;
    t.Start();
    (*_predictor)(input_vec, &output_vec);
    float fps = 1000/t.MilliSeconds();
    total_fps += fps;
    avg_fps = total_fps / iters_fps;
    total_fps -= avg_fps;

    constexpr int k = 5;
    float max[k] = {0};
    int max_index[k] = {0};
    // Find the top-k results manually.

    for (auto output : output_vec) {
        auto data = output.data<float>();
        //        std::cout << *data << ' ';

        for (auto i = 0; i < output.size(); ++i) {
            for (auto j = 0; j < k; ++j) {
                if (data[i] > max[j]) {
                    for (auto _j = k - 1; _j > j; --_j) {
                        max[_j - 1] = max[_j];
                        max_index[_j - 1] = max_index[_j];
                    }
                    max[j] = data[i];
                    max_index[j] = i;
                    goto skip;
                }
            }
            skip:;
        }
    }
    std::ostringstream stringStream;
    stringStream << avg_fps << " FPS\n";

    for (auto j = 0; j < k; ++j) {
        stringStream << j << ": " << plant_diseases_classes[max_index[j]] << " - " << max[j] / 10 << "%\n";
    }
    return env->NewStringUTF(stringStream.str().c_str());
}
cv::Mat resize_and_center_square_crop(cv::Mat input_image, int square_size=224)
{

    // Resize image so that the smallest side == square_size
    // and do a center crop along the biggest side.
    // This way we preserve the aspect ratio and prepare the image
    // for network.

    int width = input_image.cols,
            height = input_image.rows;

    int min_dim = ( width >= height ) ? height : width;
    float scale = ( ( float ) square_size ) / min_dim;

//    cv::resize(input_image, input_image, cv::Size(0, 0), scale, scale, cv::INTER_LINEAR);

    cv::Rect roi;

    if ( height >= width )
    {
        roi.width = square_size;
        roi.x = 0;

        roi.height = square_size;
        roi.y = ( input_image.rows - roi.height ) / 2;
    }
    else
    {
        roi.y = 0;
        roi.height = square_size;


        roi.width = square_size;
        roi.x = ( input_image.cols - roi.width ) / 2;
    }

    cv::Mat square_crop = input_image(roi);

    return square_crop;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_teamfuture_plantclassifier_Camera2BasicFragment_classify(JNIEnv *env, jobject instance,
                                                              jstring file_) {
    const char *file = env->GetStringUTFChars(file_, 0);
//#ifdef USE_OPENCV

//    // TODO
    cv::Mat img = cv::imread(file);
//    cv::Mat* imgAddr = (cv::Mat*) file_;
//    cv::Mat img = *imgAddr;
    CHECK(!img.empty()) << "Unable to decode image " << file_;

    cv::resize(img, img, cv::Size(256,256), cv::INTER_AREA);
    cv::Rect myRect(128 - 112, 128 - 112, 128 + 112, 128 + 112);
    img = img(myRect);

    img = resize_and_center_square_crop(img, 224);

    img = img/255.;

    img.convertTo(img, CV_32FC3);

    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);
    std::vector<float> data;

    for (auto &c : channels) {
        auto a = c.size();


//        alog("channels, %f", ((float *)c.datastart));
        data.insert(data.end(), (float *)c.datastart, (float *)c.dataend);
    }

//    cv::Vec3f intensity = img.at<cv::Vec3f>(y, x);
//    float blue = intensity.val[0];
//    float green = intensity.val[1];
//    float red = intensity.val[2];
//    auto imgA = img[:,:,0]


    // resize image


//    if (img.channels() == 3 && img.cha == 1)
//        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
//    else
//    if (img.channels() == 4 && IMG_C == 3)
//        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
//    else if (img.channels() == 1 && IMG_C == 3)
//        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
//    else
//        sample = img;

//    cv::Mat sample_resized;
//    if (sample.size() != input_geometry_)
//        cv::resize(sample, sample_resized, input_geometry_);
//    else
//        sample_resized = sample;


//    sample.convertTo(sample, CV_32FC3);
//
//    // convert NHWC to NCHW
//    std::vector<cv::Mat> channels(3);
//    cv::split(sample, channels);
//    for (auto &c : channels) {
//        alog("channels, %f", (float *)c.datastart);
//        data.insert(data.end(), (float *)c.datastart, (float *)c.dataend);
//    }

    // normalize image
    int dim = 0;
//    float imgArray[1 * IMG_C * IMG_H * IMG_W];

    float image_mean[3] = {0.485, 0.456, 0.406};
    float image_std[3] = {0.229, 0.224, 0.225};

//    for(auto i = 0; i < data.size();++i){
//        if(i > 0 && i % (224*224) == 0) dim++;
//        input_data[i] = (data[i] - image_mean[dim]) / image_std[dim];
//
//        alog("at i: %d",i);
//    }
//    float image_mean[3] = {113.865, 122.95, 125.307};
//    float image_std[3] = {66.7048, 62.0887, 62.9932};

    for(auto i = 0; i < data.size();++i){
        if(i > 0 && (i % (224*224) == 0))
            dim++;
        input_data[i] = (data[i] - image_mean[dim]) / image_std[dim];
        // std::cout << input_data[i] << std::endl;
//        alog("at i: %d, %f",i, input_data[i]);
    }
    caffe2::TensorCPU input;
    input = caffe2::Tensor(std::vector<int>({1, IMG_C, IMG_H, IMG_W}), caffe2::CPU);

    memcpy(input.mutable_data<float>(), input_data, IMG_H * IMG_W * IMG_C * sizeof(float));
    std::vector<caffe2::TensorCPU> input_vec({input});
    std::vector<caffe2::TensorCPU> output_vec;
    caffe2::Timer t;
    t.Start();
    (*_predictor)(input_vec, &output_vec);
    float fps = 1000/t.MilliSeconds();
    total_fps += fps;
    avg_fps = total_fps / iters_fps;
    total_fps -= avg_fps;

//    _predictor->operator()(input_vec, &output_vec);
//    float *rs=(float *)output_vec[0].raw_mutable_data();

//    std::vector<float> output {rs, rs + 39};

    constexpr int k = 5;
    float max[k] = {0};
    int max_index[k] = {0};

    // Find the top-k results manually.

    auto output = output_vec[2];
    auto d = (float *)output.raw_mutable_data();
//        //        std::cout << *data << ' ';

    std::vector<float> values(d, d+39);

    for (auto i = 0; i < output.size(); ++i) {
        float element = exp(d[i]);
        values[i] = element;

        alog("number %d: %f, %f  ", i, element, d[i]);
    }


    std::vector<int> V(39);
    int x=0;
    std::iota(V.begin(),V.end(),x++); //Initializing
    std::sort( V.begin(),V.end(), [&](int i,int j){return values[i] > values[j];} );

//    float sorted_idx = sort_indexes(values);

//    for (auto i: sorted_idx) {
//        cout << values[i] << endl;
//    }

//    std::sort(values.begin(), values.begin()+39, std::greater<int>());

    std::ostringstream stringStream;
    stringStream << avg_fps << " FPS\n";

    for(auto j = 0; j < k; ++j) {
        stringStream << j+1 << ": " << plant_diseases_classes[V[j]] << " - " << values[V[j]] * 100.0  << "%\n";
    }


//    for (auto i = 0; i < output.size(); ++i) {
//        float element = exp(d[i]);
//        values[i] = element;
//
//        alog("number %d: %f, %f  ",i,element, d[i]);

//        if (isinf(element)){
//            goto skip;
//        }
//        for (auto j = 0; j < k; ++j) {
//
//            if (element > max[j]) {
//                for (auto _j = k - 1; _j > j; --_j) {
//                    max[_j - 1] = max[_j];
//                    max_index[_j - 1] = max_index[_j];
//                }
//                max[j] = element;
//                max_index[j] = i;
//                goto skip;
//            }
//        }
//        skip:;
//    }
//    }

//    for (auto j = 0; j < k; ++j) {
//        stringStream << j+1 << ": " << plant_diseases_classes[max_index[j]] << " - " << max[j] * 100  << "%\n";
//    }
//    env->ReleaseStringUTFChars(file_, file);

//#endif  // USE_OPENCV

    return env->NewStringUTF(stringStream.str().c_str());
}
