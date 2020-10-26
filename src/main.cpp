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

torch::Tensor CVMatToTensor(cv::Mat mat)
{
    cv::Mat matFloat;
    mat.convertTo(matFloat, CV_32F);
    auto size = matFloat.size();
    auto tensor = torch::from_blob(matFloat.data, {size.height, size.width});
    return tensor;
}

int main()
{
    String imageName( "../area_245_0.png" );
    auto img_ = imread(samples::findFile(imageName), IMREAD_COLOR);

    if(!img_.data)
    {
        cout <<  "Could not open or find the image." << std::endl ;
        return -1;
    }

    Mat input_mat = img_.reshape(1,1);
    auto input_ = CVMatToTensor(input_mat);
    // std::cout << input_ << '\n';


    std::cout.precision(5);

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("../areas_net.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_);

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output << '\n'; // for testing output values


    if (at::is_nonzero(at::argmax(output)))
    {
        std::cout << imageName << " is dot." << '\n';
    } else
    {
        std::cout << imageName << " is area." << '\n';
    }

    return 0;
}
