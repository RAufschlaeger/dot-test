#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>

using namespace cv;
using namespace std;

int main()
{
    String imageName( "../dot_1_0.png" );
    Mat image;
    image = imread(samples::findFile(imageName), IMREAD_UNCHANGED);

    if(!image.data)
    {
        cout <<  "Could not open or find the image." << std::endl ;
        return -1;
    }

    int rows = image.rows;
    int cols = image.cols;
    cout<<rows<<" "<<cols<<endl;

    Mat bgr[3];   //destination array
    split(image,bgr);//split source

    //Note: OpenCV uses BGR color order
    //imwrite("blue.png",bgr[0]); //blue channel
    //imwrite("green.png",bgr[1]); //green channel
    //imwrite("red.png",bgr[2]); //red channel

    Mat channel;
    channel = bgr[1];

    namedWindow( "Display window", WINDOW_GUI_EXPANDED); // Create a window for display.
    imshow( "Display window", channel );                     // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window.

    Mat img;
    normalize(channel, img, 0, 1, cv::NORM_MINMAX);

    //image = image.reshape(1,1); // flat 1d
    //cv::normalize(image, image, 0, 255, NORM_L1, CV_8UC1);
    cout << "img = " << endl << " "  << img << endl << endl;


    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("../model.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    auto tensor_image = torch::from_blob(img.data, {img.rows*image.cols});
    tensor_image = tensor_image.toType(at::kFloat);

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_image);

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
