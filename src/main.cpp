#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include <iomanip>      // std::setprecision

using namespace cv;
using namespace std;

int main()
{
    String imageName( "../dot_1_0.png" );
    auto img_ = imread(samples::findFile(imageName), IMREAD_COLOR);

    if(!img_.data)
    {
        cout <<  "Could not open or find the image." << std::endl ;
        return -1;
    }

    int rows = img_.rows;
    int cols = img_.cols;
    cout<<rows<<" "<<cols<<endl;

    cv::Mat img(12, 12, CV_8UC3);
    cv::resize(img_, img, img.size(), 0, 0);
    auto input_ = torch::tensor(at::ArrayRef<uint8_t>(img.data, img.rows * img.cols * 1)).view({1, img.rows*img.cols});
    input_ = torch::flip(input_, 0);
    cout << input_ << endl << endl;
    input_ = input_.toType(at::kFloat);
    cout << input_ << endl << endl;

    std::cout.precision(5);

    Mat bgr[3]; // destination array
    split(img,bgr); // split source

    // Note: OpenCV uses BGR color order
    // imwrite("blue.png",bgr[0]); //blue channel
    // imwrite("green.png",bgr[1]); //green channel
    // imwrite("red.png",bgr[2]); //red channel

    /*
    Mat plot;
    plot = bgr[1];

    namedWindow( "Display window", WINDOW_GUI_EXPANDED); // Create a window for display.
    imshow( "Display window", image );                     // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window.


    img = bgr[2];
    img = img.reshape(1,1); // flat 1d
    cout <<  "img2 = " << endl << " "  << img << endl << endl;

    img = bgr[1];
    img = img.reshape(1,1); // flat 1d
    cout << "img1 = " << endl << " "  << img << endl << endl;

    img = bgr[0];
    // img = img.reshape(1,1); // flat 1d
    cout << "img0 = " << endl << " "  << img << endl << endl;
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    minMaxLoc( img, &minVal, &maxVal, &minLoc, &maxLoc );
    img/=maxVal;
    */


    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("../model.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    // auto tensor_image = torch::from_blob(img.data, {img.rows*img.cols});
    input_ = input_.toType(at::kFloat);
    // cout << tensor_image << endl << endl;

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_);

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output.slice(/*dim=*/0, /*start=*/0, /*end=*/2) << '\n';

    if (at::is_nonzero(at::argmax(output)))
    {
        std::cout << "Image contains dot." << '\n';
    } else
    {
        std::cout << "Image is area." << '\n';
    }


    return 0;
}
