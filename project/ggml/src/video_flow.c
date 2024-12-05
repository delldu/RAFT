/************************************************************************************
***
***	Copyright 2024 Dell Du(18588220928@163.com), All Rights Reserved.
***
***	File Author: Dell, Tue 02 Apr 2024 03:49:53 PM CST
***
************************************************************************************/

#include "raft.h"

#define GGML_ENGINE_IMPLEMENTATION
#include <ggml_engine.h>
#define GGML_NN_IMPLEMENTATION
#include <ggml_nn.h>

#include <sys/stat.h> // for chmod()


int video_flow_predict(VideoFlowNetwork *flow_net, char *input_file, char *second_file, char *output_file)
{
    TENSOR *input1_tensor, *input2_tensor, *output_tensor;

    // Loading content tensor and it's segment tensor
    {
        input1_tensor = tensor_load_image(input_file, 0 /*input_with_alpha*/);
        check_tensor(input1_tensor);

        input2_tensor = tensor_load_image(second_file, 0 /*input_with_alpha*/);
        check_tensor(input2_tensor);

    }

    // Blender input1_tensor & style_tensor
    {
        output_tensor = flow_net->forward(input1_tensor, input2_tensor);
        check_tensor(output_tensor);
        tensor_destroy(input2_tensor);
        tensor_destroy(input1_tensor);

        tensor_show("-------------- output_tensor", output_tensor);


        // TENSOR *xxxx_test;
        // xxxx_test = flow_net->net.get_output_tensor("images");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** images", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }

        // xxxx_test = flow_net->net.get_output_tensor("x1");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** x1", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }

        // xxxx_test = flow_net->net.get_output_tensor("x2");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** x2", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }

        // xxxx_test = flow_net->net.get_output_tensor("x3");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** x3", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }
        // xxxx_test = flow_net->net.get_output_tensor("x4");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** x4", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }

        // xxxx_test = flow_net->net.get_output_tensor("x5");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** x5", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }

        // xxxx_test = flow_net->net.get_output_tensor("x6");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** x6", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }

        // xxxx_test = flow_net->net.get_output_tensor("x7");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** x7", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }


        // xxxx_test = flow_net->net.get_output_tensor("fmaps");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** fmaps", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }


        // xxxx_test = flow_net->net.get_output_tensor("BatchBasicEncoder");
        // if (tensor_valid(xxxx_test)) {
        //     tensor_show("********************** BatchBasicEncoder", xxxx_test);
        //     tensor_destroy(xxxx_test);
        // }


        tensor_saveas_image(output_tensor, 0 /*batch 0*/, output_file);
        chmod(output_file, 0644);

        tensor_destroy(output_tensor);
    }

    return RET_OK;
}
