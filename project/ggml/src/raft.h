#ifndef __RAFT__H__
#define __RAFT__H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"

// --------------------------------------------------------------------------
struct FlowHead {
    struct Conv2d conv1;
    struct Conv2d conv2;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv1.in_channels = 128;
        conv1.out_channels = 256;
        conv1.kernel_size = {3, 3};
        conv1.stride = { 1, 1 };
        conv1.padding = { 1, 1 };
        conv1.create_weight_tensors(ctx);

        conv2.in_channels = 256;
        conv2.out_channels = 2;
        conv2.kernel_size = {3, 3};
        conv2.stride = { 1, 1 };
        conv2.padding = { 1, 1 };
        conv2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv1.");
        conv1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv2.");
        conv2.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // x = self.conv2(self.relu(self.conv1(x)))
        x = conv1.forward(ctx, x);
        x = ggml_relu(ctx, x);
        x = conv2.forward(ctx, x);

    	return x;
    }
};

// -------------------------------------------------------------------------------
struct SepConvGRU {
    struct Conv2d convz1;
    struct Conv2d convr1;
    struct Conv2d convq1;

    struct Conv2d convz2;
    struct Conv2d convr2;
    struct Conv2d convq2;

    void create_weight_tensors(struct ggml_context* ctx) {
        convz1.in_channels = 384;
        convz1.out_channels = 128;
        convz1.kernel_size = {1, 5};
        convz1.stride = { 1, 1 };
        convz1.padding = { 0, 2 };
        convz1.create_weight_tensors(ctx);

        convr1.in_channels = 384;
        convr1.out_channels = 128;
        convr1.kernel_size = {1, 5};
        convr1.stride = { 1, 1 };
        convr1.padding = { 0, 2 };
        convr1.create_weight_tensors(ctx);

        convq1.in_channels = 384;
        convq1.out_channels = 128;
        convq1.kernel_size = {1, 5};
        convq1.stride = { 1, 1 };
        convq1.padding = { 0, 2 };
        convq1.create_weight_tensors(ctx);

        convz2.in_channels = 384;
        convz2.out_channels = 128;
        convz2.kernel_size = {5, 1};
        convz2.stride = { 1, 1 };
        convz2.padding = { 2, 0 };
        convz2.create_weight_tensors(ctx);

        convr2.in_channels = 384;
        convr2.out_channels = 128;
        convr2.kernel_size = {5, 1};
        convr2.stride = { 1, 1 };
        convr2.padding = { 2, 0 };
        convr2.create_weight_tensors(ctx);

        convq2.in_channels = 384;
        convq2.out_channels = 128;
        convq2.kernel_size = {5, 1};
        convq2.stride = { 1, 1 };
        convq2.padding = { 2, 0 };
        convq2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "convz1.");
        convz1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "convr1.");
        convr1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "convq1.");
        convq1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "convz2.");
        convz2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "convr2.");
        convr2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "convq2.");
        convq2.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* h, ggml_tensor_t* x) {
        ggml_tensor_t *hx, *z, *r, *rh, *q, *one_z;
        // # horizontal
        hx = ggml_concat(ctx, h, x, 2 /*dim on channel*/);
        z = ggml_sigmoid(ctx, convz1.forward(ctx, hx));
        r = ggml_sigmoid(ctx, convr1.forward(ctx, hx));
        // q
        rh = ggml_mul(ctx, r, h);
        rh = ggml_concat(ctx, rh, x, 2 /*dim on channel*/);
        q = ggml_tanh(ctx, convq1.forward(ctx, rh));

        one_z = ggml_dup(ctx, z);
        // one_z = ggml_constant(ctx, one_z, 1.0);
        one_z = ggml_clamp(ctx, one_z, 1.0f, 1.0f);
        one_z = ggml_sub(ctx, one_z, z);
        h = ggml_add(ctx, ggml_mul(ctx, one_z, h), ggml_mul(ctx, z, q));

        // # vertical
        hx = ggml_concat(ctx, h, x, 2 /*dim on channel*/);
        z = ggml_sigmoid(ctx, convz2.forward(ctx, hx));
        r = ggml_sigmoid(ctx, convr2.forward(ctx, hx));
        // q
        rh = ggml_mul(ctx, r, h);
        rh = ggml_concat(ctx, rh, x, 2 /*dim on channel*/);
        q = ggml_tanh(ctx, convq2.forward(ctx, rh));
        one_z = ggml_dup(ctx, z);
        // one_z = ggml_constant(ctx, one_z, 1.0);
        one_z = ggml_clamp(ctx, one_z, 1.0f, 1.0f);
        one_z = ggml_sub(ctx, one_z, z);
        h = ggml_add(ctx, ggml_mul(ctx, one_z, h), ggml_mul(ctx, z, q));

    	return h;
    }
};

// -----------------------------------------------------------
struct MotionEncoder {
    struct Conv2d convc1;
    struct Conv2d convc2;
    struct Conv2d convf1;
    struct Conv2d convf2;
    struct Conv2d conv;

    void create_weight_tensors(struct ggml_context* ctx) {
        convc1.in_channels = 324;
        convc1.out_channels = 256;
        convc1.kernel_size = {1, 1};
        convc1.stride = { 1, 1 };
        convc1.padding = { 0, 0 };
        convc1.create_weight_tensors(ctx);

        convc2.in_channels = 256;
        convc2.out_channels = 192;
        convc2.kernel_size = {3, 3};
        convc2.stride = { 1, 1 };
        convc2.padding = { 1, 1 };
        convc2.create_weight_tensors(ctx);

        convf1.in_channels = 2;
        convf1.out_channels = 128;
        convf1.kernel_size = {7, 7};
        convf1.stride = { 1, 1 };
        convf1.padding = { 3, 3 };
        convf1.create_weight_tensors(ctx);

        convf2.in_channels = 128;
        convf2.out_channels = 64;
        convf2.kernel_size = {3, 3};
        convf2.stride = { 1, 1 };
        convf2.padding = { 1, 1 };
        convf2.create_weight_tensors(ctx);

        // (conv): Conv2d(256, 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        conv.in_channels = 256;
        conv.out_channels = 126;
        conv.kernel_size = {3, 3};
        conv.stride = { 1, 1 };
        conv.padding = { 1, 1 };
        conv.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "convc1.");
        convc1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "convc2.");
        convc2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "convf1.");
        convf1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "convf2.");
        convf2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* flow, ggml_tensor_t* corr) {
        // cor = F.relu(self.convc1(corr))
        // cor = F.relu(self.convc2(cor))
        // flo = F.relu(self.convf1(flow))
        // flo = F.relu(self.convf2(flo))

        // cor_flo = torch.cat([cor, flo], dim=1)
        // out = F.relu(self.conv(cor_flo))
        // motion_feat = torch.cat([out, flow], dim=1)

        // # tensor [motion_feat] size: [1, 128, 55, 128], min: -1.508427, max: 6.19675, mean: 0.129374
        // return motion_feat
        ggml_tensor_t *cor;
        cor = convc1.forward(ctx, corr);
        cor = ggml_relu(ctx, cor);
        cor = convc2.forward(ctx, cor);
        cor = ggml_relu(ctx, cor);

        ggml_tensor_t *flo;
        flo = convf1.forward(ctx, flow);
        flo = ggml_relu(ctx, flo);
        flo = convf2.forward(ctx, flo);
        flo = ggml_relu(ctx, flo);

        ggml_tensor_t *cor_flo;
        cor_flo = ggml_concat(ctx, cor, flo, 2 /*dim on channel*/);
        cor_flo = conv.forward(ctx, cor_flo);
        cor_flo = ggml_relu(ctx, cor_flo);

        ggml_tensor_t *motion_feat;
        motion_feat = ggml_concat(ctx, cor_flo, flow, 2/*dim on channel*/);

        return motion_feat;
    }
};

struct BasicUpdateBlock {
    struct MotionEncoder encoder;
    struct SepConvGRU gru;
    struct FlowHead flow_head;
    struct Conv2d mask_0;
    struct Conv2d mask_2;

    void create_weight_tensors(struct ggml_context* ctx) {
        encoder.create_weight_tensors(ctx);
        gru.create_weight_tensors(ctx);
        flow_head.create_weight_tensors(ctx);

        mask_0.in_channels = 128;
        mask_0.out_channels = 256;
        mask_0.kernel_size = {3, 3};
        mask_0.stride = { 1, 1 };
        mask_0.padding = { 1, 1 };
        mask_0.create_weight_tensors(ctx);

        mask_2.in_channels = 256;
        mask_2.out_channels = 576;
        mask_2.kernel_size = {1, 1};
        mask_2.stride = { 1, 1 };
        mask_2.padding = { 0, 0 };
        mask_2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "encoder.");
        encoder.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "gru.");
        gru.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "flow_head.");
        flow_head.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "mask.0.");
        mask_0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "mask.2.");
        mask_2.setup_weight_names(s);
    }


    // def forward(self, net, inp, corr, flow) -> List[torch.Tensor]:
    //     motion_feat = self.encoder(flow, corr)
    //     inp = torch.cat([inp, motion_feat], dim=1)

    //     net = self.gru(net, inp)
    //     delta = self.flow_head(net)

    //     # scale mask to balence gradients
    //     mask = 0.25 * self.mask(net)
    //     return net, mask, delta
    std::vector<ggml_tensor_t *> forward(struct ggml_context* ctx, 
        ggml_tensor_t* net, ggml_tensor_t* inp, ggml_tensor_t* corr, ggml_tensor_t* flow) {
        // tensor [net] size: [1, 128, 55, 128], min: -1.0, max: 1.0, mean: -0.000222
        // tensor [inp] size: [1, 128, 55, 128], min: 0.0, max: 5.488075, mean: 0.038178
        // tensor [corr] size: [1, 324, 55, 128], min: -5.405947, max: 25.431339, mean: 0.233413
        // tensor [flow] size: [1, 2, 55, 128], min: 0.0, max: 0.0, mean: 0.0

        std::vector<ggml_tensor_t *> net_mask_flow_list;
        {
            // ggml_tensor_t *x2 = ggml_dup(ctx, flow);
            // x2 = ggml_cont(ctx, x2);
            // ggml_set_output(x2);
            // ggml_set_name(x2, "x_flow");

            corr = ggml_cont(ctx, corr);
            ggml_set_name(corr, "x_corr");
            ggml_set_output(corr);
        }

        // xxxx_debug_1111
        ggml_tensor_t *motion_feat = encoder.forward(ctx, flow, corr);
        {
            motion_feat = ggml_cont(ctx, motion_feat);
            ggml_set_name(motion_feat, "x_motion_feat");
            ggml_set_output(motion_feat);
        }

        ggml_tensor_t *inp_out = ggml_concat(ctx, inp, motion_feat, 2 /*dim on channe*/);
        {
            inp_out = ggml_cont(ctx, inp_out);
            ggml_set_name(inp_out, "x_inp");
            ggml_set_output(inp_out);
        }

        ggml_tensor_t *net_out = gru.forward(ctx, net, inp_out);
        {
            net_out = ggml_cont(ctx, net_out);
            ggml_set_name(net_out, "x_net");
            ggml_set_output(net_out);
        }

        ggml_tensor_t *delta = flow_head.forward(ctx, net_out);
        {
            delta = ggml_cont(ctx, delta);
            ggml_set_name(delta, "x_delta_flow");
            ggml_set_output(delta);
        }


        // 1) 1
        // Info: Warnning: 'x_flow' NOT found in output tensors.
        // Info: ********************** x_corr Tensor: 1x324x55x128
        // min: -4.7871, max: 25.4170, mean: 0.5577
        // Info: ********************** x_motion_feat Tensor: 1x128x55x128
        // min: 0.0000, max: 12.0330, mean: 0.1776
        // Info: ********************** x_inp Tensor: 1x2x55x128
        // min: 0.0000, max: 0.0000, mean: 0.0000
        // Info: ********************** x_net Tensor: 1x128x55x128
        // min: -1.0000, max: 1.0000, mean: -0.0076
        // Info: ********************** x_delta_flow Tensor: 1x2x55x128
        // min: -7.8906, max: 4.9102, mean: -0.2951
        // --------------------------------------------------------------------------------
        // tensor [x_flow] size: [1, 2, 55, 128], min: 0.0, max: 0.0, mean: 0.0
        // tensor [x_corr] size: [1, 324, 55, 128], min: -5.405947, max: 25.431339, mean: 0.315328
        // tensor [x_motion_feat] size: [1, 128, 55, 128], min: 0.0, max: 28.339005, mean: 0.249131
        // tensor [x_inp] size: [1, 256, 55, 128], min: 0.0, max: 28.339005, mean: 0.143654
        // tensor [x_net] size: [1, 128, 55, 128], min: -1.0, max: 1.0, mean: 0.005785
        // tensor [x_delta_flow] size: [1, 2, 55, 128], min: -1.455234, max: 1.324562, mean: -0.113951
        // --------------------------------------------------------------------------------

        //xxxx_debug_1111
        // 2) 10
        // Info: Warnning: 'x_flow' NOT found in output tensors.
        // Info: ********************** x_corr Tensor: 1x324x55x128
        // min: -4.3440, max: 17.3228, mean: 0.3212
        // Info: ********************** x_motion_feat Tensor: 1x128x55x128
        // min: -65.4738, max: 62.8011, mean: 0.1026
        // Info: ********************** x_inp Tensor: 1x2x55x128
        // min: -65.4738, max: 62.8011, mean: -0.0175
        // Info: ********************** x_net Tensor: 1x128x55x128
        // min: -1.0000, max: 1.0000, mean: 0.0121
        // Info: ********************** x_delta_flow Tensor: 1x2x55x128
        // min: -12.0623, max: 12.0313, mean: 0.0079
        // --------------------------------------------------------------------------------
        // tensor [x_flow] size: [1, 2, 55, 128], min: -1.266312, max: 1.017403, mean: -0.111345
        // tensor [x_corr] size: [1, 324, 55, 128], min: -4.746385, max: 25.08046, mean: 0.316789
        // tensor [x_motion_feat] size: [1, 128, 55, 128], min: -1.266312, max: 10.188622, mean: 0.13014
        // tensor [x_inp] size: [1, 256, 55, 128], min: -1.266312, max: 10.188622, mean: 0.084159
        // tensor [x_net] size: [1, 128, 55, 128], min: -1.0, max: 1.0, mean: 0.056684
        // tensor [x_delta_flow] size: [1, 2, 55, 128], min: -0.15567, max: 0.179367, mean: 8.9e-05
        // --------------------------------------------------------------------------------

        // 2) 20
        // Info: Warnning: 'x_flow' NOT found in output tensors.
        // Info: ********************** x_corr Tensor: 1x324x55x128
        // min: -5.0586, max: 16.8043, mean: 0.2397
        // Info: ********************** x_motion_feat Tensor: 1x128x55x128
        // min: -175.0705, max: 178.6842, mean: 0.1930
        // Info: ********************** x_inp Tensor: 1x2x55x128
        // min: -175.0705, max: 178.6842, mean: -0.0618
        // Info: ********************** x_net Tensor: 1x128x55x128
        // min: -1.0000, max: 1.0000, mean: 0.0123
        // Info: ********************** x_delta_flow Tensor: 1x2x55x128
        // min: -11.5233, max: 13.1407, mean: -0.0241
        // --------------------------------------------------------------------------------
        // tensor [x_flow] size: [1, 2, 55, 128], min: -1.243141, max: 1.013954, mean: -0.111469
        // tensor [x_corr] size: [1, 324, 55, 128], min: -4.747513, max: 24.461765, mean: 0.31686
        // tensor [x_motion_feat] size: [1, 128, 55, 128], min: -1.243141, max: 10.150092, mean: 0.130334
        // tensor [x_inp] size: [1, 256, 55, 128], min: -1.243141, max: 10.150092, mean: 0.084256
        // tensor [x_net] size: [1, 128, 55, 128], min: -1.0, max: 1.0, mean: 0.053441
        // tensor [x_delta_flow] size: [1, 2, 55, 128], min: -0.134227, max: 0.147832, mean: -1e-05
        // --------------------------------------------------------------------------------



        // # scale mask to balence gradients
        // mask = 0.25 * self.mask(net)
        ggml_tensor_t *mask;
        mask = mask_0.forward(ctx, net_out);
        mask = ggml_relu(ctx, mask);
        mask = mask_2.forward(ctx, mask);
        mask = ggml_scale(ctx, mask, 0.25f);

        net_mask_flow_list.push_back(net_out);
        net_mask_flow_list.push_back(mask);
        net_mask_flow_list.push_back(delta);

    	return net_mask_flow_list;
    }
};


// ResidualBlock
struct BatchResidualBlock {
    int in_planes;
    int planes;
    int stride = 1;

    // network hparams
    struct Conv2d conv1;
    struct Conv2d conv2;

    struct BatchNorm2d norm1;
    struct BatchNorm2d norm2;

    struct Conv2d downsample_conv;
    struct BatchNorm2d downsample_norm;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv1.in_channels = in_planes;
        conv1.out_channels = planes;
        conv1.kernel_size = {3, 3};
        conv1.stride = { stride, stride };
        conv1.padding = { 1, 1 };
        conv1.create_weight_tensors(ctx);

        conv2.in_channels = planes;
        conv2.out_channels = planes;
        conv2.kernel_size = {3, 3};
        conv2.stride = { 1, 1 };
        conv2.padding = { 1, 1 };
        conv2.create_weight_tensors(ctx);

        norm1.num_features = planes;
        norm1.create_weight_tensors(ctx);

        norm2.num_features = planes;
        norm2.create_weight_tensors(ctx);

        if (stride > 1) {
            // for downsample ...
            // nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride)        
            downsample_conv.in_channels = in_planes;
            downsample_conv.out_channels = planes;
            downsample_conv.kernel_size = {1, 1};
            downsample_conv.stride = { stride, stride };
            downsample_conv.padding = { 0, 0 };
            downsample_conv.create_weight_tensors(ctx);

            downsample_norm.num_features = planes;
            downsample_norm.create_weight_tensors(ctx);
        }
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv1.");
        conv1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "conv2.");
        conv2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "norm2.");
        norm2.setup_weight_names(s);

        if (stride > 1) {
             // for downsample ...
            snprintf(s, sizeof(s), "%s%s", prefix, "downsample.0.");
            downsample_conv.setup_weight_names(s);
            snprintf(s, sizeof(s), "%s%s", prefix, "downsample.1.");
            downsample_norm.setup_weight_names(s);
        }
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // y = x
        // y = self.relu(self.norm1(self.conv1(y)))
        // y = self.relu(self.norm2(self.conv2(y)))
        // x = self.downsample(x)
        // return self.relu(x + y)

        ggml_tensor_t *y = x;
        y = conv1.forward(ctx, y);
        y = norm1.forward(ctx, y);
        y = ggml_relu(ctx, y);
        y = conv2.forward(ctx, y);
        y = norm2.forward(ctx, y);
        y = ggml_relu(ctx, y);
        if (stride > 1) {
            x = downsample_conv.forward(ctx, x);
            x = downsample_norm.forward(ctx, x);
        }
        x = ggml_relu(ctx, ggml_add(ctx, x, y));

    	return x;
    }
};

// ResidualBlock
// --------------------------------------------------------
struct InstanceResidualBlock {
    int in_planes;
    int planes;
    int stride = 1;

    // network hparams
    struct Conv2d conv1;
    struct Conv2d conv2;

    struct InstanceNorm2d norm1;
    struct InstanceNorm2d norm2;

    struct Conv2d downsample_conv;
    struct InstanceNorm2d downsample_norm;

    void create_weight_tensors(struct ggml_context* ctx) {
        // self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        // self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)

        conv1.in_channels = in_planes;
        conv1.out_channels = planes;
        conv1.kernel_size = {3, 3};
        conv1.stride = { stride, stride };
        conv1.padding = { 1, 1 };
        conv1.create_weight_tensors(ctx);

        conv2.in_channels = planes;
        conv2.out_channels = planes;
        conv2.kernel_size = {3, 3};
        conv2.stride = { 1, 1 };
        conv2.padding = { 1, 1 };
        conv2.create_weight_tensors(ctx);

        norm1.num_features = planes;
        norm1.create_weight_tensors(ctx);

        norm2.num_features = planes;
        norm2.create_weight_tensors(ctx);

        if (stride > 1) {
            // for downsample ...
            // nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride)        
            downsample_conv.in_channels = in_planes;
            downsample_conv.out_channels = planes;
            downsample_conv.kernel_size = {1, 1};
            downsample_conv.stride = { stride, stride };
            downsample_conv.padding = { 0, 0 };
            downsample_conv.create_weight_tensors(ctx);

            downsample_norm.num_features = planes;
            downsample_norm.create_weight_tensors(ctx);
        }
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv1.");
        conv1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "conv2.");
        conv2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "norm2.");
        norm2.setup_weight_names(s);

        if (stride > 1) {
            // for downsample ...
            snprintf(s, sizeof(s), "%s%s", prefix, "downsample.0.");
            downsample_conv.setup_weight_names(s);
            snprintf(s, sizeof(s), "%s%s", prefix, "downsample.1.");
            downsample_norm.setup_weight_names(s);
        }
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // y = x
        // y = self.relu(self.norm1(self.conv1(y)))
        // y = self.relu(self.norm2(self.conv2(y)))
        // x = self.downsample(x)
        // return self.relu(x + y)
        ggml_tensor_t *y = x;
        y = conv1.forward(ctx, y);
        y = norm1.forward(ctx, y);
        y = ggml_relu(ctx, y);
        y = conv2.forward(ctx, y);
        y = norm2.forward(ctx, y);
        y = ggml_relu(ctx, y);
        if (stride > 1) { // downsample
            x = downsample_conv.forward(ctx, x);
            x = downsample_norm.forward(ctx, x);
        }
        x = ggml_relu(ctx,  ggml_add(ctx, x, y)); // self.relu(x + y)

        return x;
    }
};

// BasicEncoder
struct BatchBasicEncoder {
    struct BatchNorm2d norm1;

    struct Conv2d conv1;

    struct BatchResidualBlock layer1_0;
    struct BatchResidualBlock layer1_1;
    struct BatchResidualBlock layer2_0;
    struct BatchResidualBlock layer2_1;
    struct BatchResidualBlock layer3_0;
    struct BatchResidualBlock layer3_1;

    struct Conv2d conv2;

    void create_weight_tensors(struct ggml_context* ctx) {
        norm1.num_features = 64;
        norm1.create_weight_tensors(ctx);

        // self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        conv1.in_channels = 3;
        conv1.out_channels = 64;
        conv1.kernel_size = {7, 7};
        conv1.stride = { 2, 2 };
        conv1.padding = { 3, 3 };
        conv1.create_weight_tensors(ctx);

        layer1_0.in_planes = 64;
        layer1_0.planes = 64;
        layer1_0.stride = 1;
        layer1_0.create_weight_tensors(ctx);

        layer1_1.in_planes = 64;
        layer1_1.planes = 64;
        layer1_1.stride = 1;
        layer1_1.create_weight_tensors(ctx);

        // self.layer2 = self._make_layer(96, stride=2)
        layer2_0.in_planes = 64;
        layer2_0.planes = 96;
        layer2_0.stride = 2;
        layer2_0.create_weight_tensors(ctx);

        layer2_1.in_planes = 96;
        layer2_1.planes = 96;
        layer2_1.stride = 1;
        layer2_1.create_weight_tensors(ctx);

        // self.layer3 = self._make_layer(128, stride=2)
        layer3_0.in_planes = 96;
        layer3_0.planes = 128;
        layer3_0.stride = 2;
        layer3_0.create_weight_tensors(ctx);

        layer3_1.in_planes = 128;
        layer3_1.planes = 128;
        layer3_1.stride = 1;
        layer3_1.create_weight_tensors(ctx);

        // self.conv2 = nn.Conv2d(128, 256, kernel_size=1)
        conv2.in_channels = 128;
        conv2.out_channels = 256;
        conv2.kernel_size = {1, 1};
        conv2.stride = { 1, 1 };
        conv2.padding = { 0, 0 };
        conv2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv1.");
        conv1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "layer1.0.");
        layer1_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer1.1.");
        layer1_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer2.0.");
        layer2_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer2.1.");
        layer2_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.0.");
        layer3_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.1.");
        layer3_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv2.");
        conv2.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // x = self.conv1(x)
        // x = self.norm1(x)
        // x = self.relu1(x)

        // x = self.layer1(x)
        // x = self.layer2(x)
        // x = self.layer3(x)

        // x = self.conv2(x)
        // return x

        x = conv1.forward(ctx, x);
        x = norm1.forward(ctx, x);
        x = ggml_relu(ctx, x);

        x = layer1_0.forward(ctx, x);
        x = layer1_1.forward(ctx, x);
        x = layer2_0.forward(ctx, x);
        x = layer2_1.forward(ctx, x);
        x = layer3_0.forward(ctx, x);
        x = layer3_1.forward(ctx, x);

        x = conv2.forward(ctx, x);
    	return x;
    }
};

// BasicEncoder
struct InstanceBasicEncoder {
    struct InstanceNorm2d norm1;

    struct Conv2d conv1;

    struct InstanceResidualBlock layer1_0;
    struct InstanceResidualBlock layer1_1;
    struct InstanceResidualBlock layer2_0;
    struct InstanceResidualBlock layer2_1;
    struct InstanceResidualBlock layer3_0;
    struct InstanceResidualBlock layer3_1;

    struct Conv2d conv2;

    void create_weight_tensors(struct ggml_context* ctx) {
        norm1.num_features = 64;
        norm1.create_weight_tensors(ctx);

        // self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        conv1.in_channels = 3;
        conv1.out_channels = 64;
        conv1.kernel_size = {7, 7};
        conv1.stride = { 2, 2 };
        conv1.padding = { 3, 3 };
        conv1.create_weight_tensors(ctx);

        layer1_0.in_planes = 64;
        layer1_0.planes = 64;
        layer1_0.stride = 1;
        layer1_0.create_weight_tensors(ctx);

        layer1_1.in_planes = 64;
        layer1_1.planes = 64;
        layer1_1.stride = 1;
        layer1_1.create_weight_tensors(ctx);

        layer2_0.in_planes = 64;
        layer2_0.planes = 96;
        layer2_0.stride = 2;
        layer2_0.create_weight_tensors(ctx);

        layer2_1.in_planes = 96;
        layer2_1.planes = 96;
        layer2_1.stride = 1;
        layer2_1.create_weight_tensors(ctx);

        layer3_0.in_planes = 96;
        layer3_0.planes = 128;
        layer3_0.stride = 2;
        layer3_0.create_weight_tensors(ctx);

        layer3_1.in_planes = 128;
        layer3_1.planes = 128;
        layer3_1.stride = 1;
        layer3_1.create_weight_tensors(ctx);

        conv2.in_channels = 128;
        conv2.out_channels = 256;
        conv2.kernel_size = {1, 1};
        conv2.stride = { 1, 1 };
        conv2.padding = { 0, 0 };
        conv2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv1.");
        conv1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "layer1.0.");
        layer1_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer1.1.");
        layer1_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer2.0.");
        layer2_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer2.1.");
        layer2_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.0.");
        layer3_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer3.1.");
        layer3_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv2.");
        conv2.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // x = self.conv1(x)
        // x = self.norm1(x)
        // x = self.relu1(x)

        // x = self.layer1(x)
        // x = self.layer2(x)
        // x = self.layer3(x)

        // x = self.conv2(x)
        // return x
        x = conv1.forward(ctx, x);
        x = norm1.forward(ctx, x);
        x = ggml_relu(ctx, x);

        x = layer1_0.forward(ctx, x);
        x = layer1_1.forward(ctx, x);

        x = layer2_0.forward(ctx, x);
        x = layer2_1.forward(ctx, x);

        x = layer3_0.forward(ctx, x);
        x = layer3_1.forward(ctx, x);

        x = conv2.forward(ctx, x);

        return x;
    }
};

struct RAFT : GGMLNetwork {
    int MAX_H = 1024;
    int MAX_W = 1024;
    int MAX_TIMES = 8;

    // network params
    struct BatchBasicEncoder cnet;
    struct InstanceBasicEncoder fnet;
    struct BasicUpdateBlock update_block;
    ggml_tensor_t *mesh_grid_9x9;

    size_t get_graph_size()
    {
        return GGML_DEFAULT_GRAPH_SIZE * 4; // 2048 * 4
    }

    void create_weight_tensors(struct ggml_context* ctx) {
        cnet.create_weight_tensors(ctx);
        fnet.create_weight_tensors(ctx);
        update_block.create_weight_tensors(ctx);

        mesh_grid_9x9 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 9, 9, 1);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "cnet.");
        cnet.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "fnet.");
        fnet.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "update_block.");
        update_block.setup_weight_names(s);

        ggml_format_name(mesh_grid_9x9, "%s%s", prefix, "mesh_grid_9x9");
    }

    ggml_tensor_t* resize_pad(struct ggml_context* ctx, ggml_tensor_t* x) {
        int W = (int)x->ne[0];
        int H = (int)x->ne[1];

        if (H > MAX_H || W > MAX_W) { // need resize ?
            float s = (float)MAX_H/H;
            if (s < (float)MAX_W/W) {
                s = (float)MAX_W/W;
            }
            int SH = s * H; // new width
            int SW = s * W; // new height
            x = ggml_interpolate(ctx, x, 1 /*dim on H */, SH);
            x = ggml_interpolate(ctx, x, 0 /*dim on W */, SW);
        }

        // Need pad ?        
        W = (int)x->ne[0];
        H = (int)x->ne[1];
        int r_pad = (MAX_TIMES - (W % MAX_TIMES)) % MAX_TIMES;
        int l_pad = r_pad/2; r_pad = r_pad - l_pad;
        int b_pad = (MAX_TIMES - (H % MAX_TIMES)) % MAX_TIMES;
        int t_pad = b_pad/2; b_pad = b_pad - t_pad;

        if (l_pad > 0 || r_pad > 0 || t_pad > 0 || b_pad > 0) {
            x = ggml_replication_pad2d(ctx, x, l_pad, r_pad, t_pad, b_pad);
        }

        return x;
    }

    // ---------------------------------------------------------------------------------------------
    ggml_tensor_t* bilinear_sampler(struct ggml_context* ctx, ggml_tensor_t* one_layer_corr, ggml_tensor_t* coords) {
        // tensor [img] size: [7040, 1, 55, 128], min: -7.369149, max: 25.431339, mean: 0.033188
        // tensor [coords] size: [7040, 9, 9, 2], min: -4.0, max: 131.0, mean: 45.25
        int W = (int)one_layer_corr->ne[0];
        int H = (int)one_layer_corr->ne[1];

        ggml_tensor_t * mesh_x, *mesh_y;
        mesh_x = ggml_nn_slice(ctx, coords, 0 /*dim*/, 0 /*start*/, 1 /*stop*/, 1/*step*/);
        mesh_y = ggml_nn_slice(ctx, coords, 0 /*dim*/, 1 /*start*/, 2 /*stop*/, 1/*step*/);
        mesh_x = ggml_scale(ctx, mesh_x, 2.0f/(W - 1.0f));
        mesh_x = ggml_add_constant(ctx, mesh_x, -1.0);
        mesh_y = ggml_scale(ctx, mesh_y, 2.0f/(H - 1.0f));
        mesh_y = ggml_add_constant(ctx, mesh_y, -1.0);
        ggml_tensor_t *grid = ggml_concat(ctx, mesh_x, mesh_y, 0/*dim on batch*/);

        grid = ggml_clamp(ctx, grid, -1.0, 1.0); // meet ggml_grid_sample spec !!!
        one_layer_corr = ggml_grid_sample(ctx, one_layer_corr, grid);

        return one_layer_corr; // f32 [9, 9, 1, 7040]
    }

    std::vector<ggml_tensor_t *> create_corr_pyramid(struct ggml_context* ctx, ggml_tensor_t* fmap1, ggml_tensor_t* fmap2) {
        std::vector<ggml_tensor_t *> corr_pyramid_list;
        int W = (int)fmap1->ne[0];
        int H = (int)fmap1->ne[1];
        int C = (int)fmap1->ne[2];
        int B = (int)fmap1->ne[3];

        fmap1 = ggml_reshape_3d(ctx, fmap1, W*H, C, B);
        fmap2 = ggml_reshape_3d(ctx, fmap2, W*H, C, B);

        fmap1 = ggml_cont(ctx, ggml_transpose(ctx, fmap1));
        ggml_tensor_t *corr = ggml_nn_mul_mat(ctx, fmap1, fmap2);
        corr = ggml_reshape_4d(ctx, corr, W, H, 1, H*W);
        corr = ggml_scale(ctx, corr, 1.0f/16.0);

        corr_pyramid_list.push_back(corr);
        for (int i = 0; i < 3; i++) {
            corr = ggml_pool_2d(ctx, corr, GGML_OP_POOL_AVG, 2 /*k0*/, 2/*k1*/, 2 /*s0*/, 2/*s1*/, 0.0/*p0*/, 0.0/*p1*/);
            corr_pyramid_list.push_back(corr);
        }

        // # corr_pyramid_list is list: len = 4
        // #     tensor [item] size: [7040, 1, 55, 128], min: -7.369149, max: 25.431339, mean: 0.033188
        // #     tensor [item] size: [7040, 1, 27, 64], min: -3.66336, max: 9.582128, mean: 0.032375
        // #     tensor [item] size: [7040, 1, 13, 32], min: -2.107447, max: 4.198452, mean: 0.03262
        // #     tensor [item] size: [7040, 1, 6, 16], min: -1.357178, max: 2.21133, mean: 0.03297
        return corr_pyramid_list;
    }

    // sample corrs
    ggml_tensor_t * index_corr_volume(struct ggml_context* ctx, ggml_tensor_t *coords, std::vector<ggml_tensor_t *>corr_pyramid) {
        ggml_tensor_t *centroid, *corr, *out_pyramid[4];
        // # tensor [coords] size: [1, 2, 55, 128], min: 0.0, max: 127.0, mean: 45.25
        int W = (int)coords->ne[0];
        int H = (int)coords->ne[1];
        int D = (int)coords->ne[2];
        int B = (int)coords->ne[3];
        coords = ggml_cont(ctx, ggml_permute(ctx, coords, 1, 2, 0, 3)); // [W, H, 2, B] --> [2, W, H, B]
        coords = ggml_reshape_4d(ctx, coords, 2, 1, 1, B*H*W);
        // tensor [coords] size: [1, 55, 128, 2], min: 0.0, max: 127.0, mean: 45.25
        // tensor [mesh_grid_9x9] size: [1, 9, 9, 2], min: -4.0, max: 4.0, mean: 0.0

        float fscale = 1.0f;
        for (int i = 0; i < 4; i++) {
            centroid = ggml_scale(ctx, coords, fscale); fscale /= 2.0f;
            centroid = ggml_repeat_ext(ctx, centroid, 1, 9, 9, 1);
            centroid = ggml_add(ctx, centroid, mesh_grid_9x9);
            // tensor [centroid] size: [7040, 1, 1, 2], min: 0.0, max: 15.875, mean: 5.65625
 
            corr = bilinear_sampler(ctx, corr_pyramid[i], centroid);
            corr = ggml_reshape_4d(ctx, corr, -1, W, H, B);
            out_pyramid[i] = corr;
        }

        ggml_tensor_t *out = ggml_cat(ctx, 4, out_pyramid[0], out_pyramid[1], out_pyramid[2], out_pyramid[3], 0/*dim*/);
        out = ggml_cont(ctx, ggml_permute(ctx, out, 2, 0, 1, 3)); // [C, W, H, B] -> [W, H, C, B]
        // tensor [out] size: [1, 324, 55, 128], min: -3.942287, max: 6.722563, mean: 0.003053

        return out;
    }

    ggml_tensor_t* upsample_flow(ggml_context_t* ctx, ggml_tensor_t *flow, ggml_tensor_t *mask) {
        int B = (int)flow->ne[3];
        int D = (int)flow->ne[2];
        int H = (int)flow->ne[1];
        int W = (int)flow->ne[0];
        // flow    f32 [128, 55, 2, 1], 

        ggml_tensor_t *final_flow;
        flow = ggml_scale(ctx, flow, 8.0);
        final_flow = ggml_nn_unfold(ctx, flow, 3 /*k0*/, 3 /*k1*/, 1 /*s0*/, 1/*s1*/, 1/*p0*/, 1/*p1*/, 1/*d0*/, 1/*d1*/);
        // [128*55, 18, 1]

        mask = ggml_reshape_4d(ctx, mask, H*W, 64, 9, 1);
        mask = ggml_softmax(ctx, mask, 2 /*dim*/);
        mask = ggml_repeat_ext(ctx, mask, 1, 1, 1, 2);

        final_flow = ggml_reshape_4d(ctx, final_flow, H*W, 1, 9, 2);
        final_flow = ggml_repeat_ext(ctx, final_flow, 1, 64, 1, 1);
        final_flow = ggml_mul(ctx, mask, final_flow); // mask * final_flow

        // up_flow = torch.sum(mask * up_flow, dim=1)
        final_flow = ggml_mean_ext(ctx, final_flow, 2 /*dim*/);
        final_flow = ggml_scale(ctx, final_flow, 9.0f);

        final_flow = ggml_reshape_4d(ctx, final_flow, W, H, 64, 2);
        final_flow = ggml_shuffle(ctx, final_flow, 8);
        final_flow = ggml_reshape_4d(ctx, final_flow, 8*W, 8*H, 2, B);

        return final_flow;
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[]) {
        int W, H, C, B;
        std::vector<ggml_tensor_t *>corr_pyramid;
        ggml_tensor_t *image1, *image2, *mask, *flow, *net, *inp;

        GGML_UNUSED(argc);
        image1 = argv[0];
        image2 = argv[1];

        mesh_grid_9x9 = ggml_cont(ctx, ggml_permute(ctx, mesh_grid_9x9, 3, 1, 2, 0)); // [2, 9, 9, 1] -> [1, 9, 9, 2]

        {
            // image1 = self.resize_pad(image1)
            // image2 = self.resize_pad(image2)
            // image1 = 2 * image1 - 1.0
            // image2 = 2 * image2 - 1.0
            image1 = resize_pad(ctx, image1);
            image2 = resize_pad(ctx, image2);
            W = (int)image1->ne[0];
            H = (int)image1->ne[1];
            C = (int)image1->ne[2];
            B = (int)image1->ne[3];

            image1 = ggml_scale(ctx, image1, 2.0);
            image1 = ggml_add_constant(ctx, image1, -1.0);

            image2 = ggml_scale(ctx, image2, 2.0);
            image2 = ggml_add_constant(ctx, image2, -1.0);
        }

        {
            ggml_tensor_t *images, *fmaps, *fmap1, *fmap2;
            images = ggml_concat(ctx, image1, image2, 3 /*dim on Batch*/);

            fmaps = fnet.forward(ctx, images);
            fmap1 = ggml_nn_slice(ctx, fmaps, 3/*dim*/, 0/*start*/, 1/*stop*/, 1/*step*/);
            fmap2 = ggml_nn_slice(ctx, fmaps, 3/*dim*/, 1/*start*/, 2/*stop*/, 1/*step*/);
            corr_pyramid = create_corr_pyramid(ctx, fmap1, fmap2);
        }

        {
            ggml_tensor_t *cnet_out = cnet.forward(ctx, image1);
            // # tensor [cnet_out] size: [1, 256, 55, 128], min: -17.80987, max: 14.065307, mean: -0.649572

            int N = (int)cnet_out->ne[2]/2;
            net = ggml_nn_slice(ctx, cnet_out, 2/*dim on channel*/, 0*N, 1*N, 1/*step*/);
            inp = ggml_nn_slice(ctx, cnet_out, 2/*dim on channel*/, 1*N, 2*N, 1/*step*/);

            net = ggml_tanh(ctx, net);
            inp = ggml_relu(ctx, inp);

            // cnet_out = ggml_cont(ctx, cnet_out);
            // ggml_set_name(cnet_out, "cnet");
            // ggml_set_output(cnet_out);

            // net = ggml_cont(ctx, net);
            // ggml_set_name(net, "net");
            // ggml_set_output(net);

            // inp = ggml_cont(ctx, inp);
            // ggml_set_name(net, "inp");
            // ggml_set_output(inp);

            // # tensor [cnet] size: [1, 256, 55, 128], min: -17.80987, max: 14.065307, mean: -0.649572
        }


        // --------------------------------------------------------------------------------------------------------
        {
            ggml_tensor_t *corr, *delta, *coords0, *coords1;
            std::vector<ggml_tensor_t *>net_mask_flow_list;

            coords0 = ggml_grid_mesh(ctx, B, H/8, W/8, 0/*norm*/);
            coords0 = ggml_cont(ctx, ggml_permute(ctx, coords0, 2, 0, 1, 3)); // [2, W, H, B] --> [W, H, 2, B]
            coords1 = ggml_dup(ctx, coords0);
            // # tensor [coords0] size: [1, 2, 55, 128], min: 0.0, max: 127.0, mean: 45.25

            flow = ggml_sub(ctx, coords1, coords0);
            for (int i = 0; i < 20; i++) { // xxxx_debug_1111
                corr = index_corr_volume(ctx, coords1, corr_pyramid);
                // tensor [corr] size: [1, 324, 55, 128], min: -4.747513, max: 24.461765, mean: 0.31686

                // corr = ggml_cont(ctx, corr);
                // ggml_set_name(corr, "corr");
                // ggml_set_output(corr);

                // CUDA Info: ********************** corr Tensor: 1x324x55x128
                // min: -5.0586, max: 16.8043, mean: 0.2397
                // CPU Info: ********************** corr Tensor: 1x324x55x128
                // min: -4.3095, max: 18.2056, mean: 0.3741

                net_mask_flow_list = update_block.forward(ctx, net, inp, corr, flow);
                net = net_mask_flow_list[0]; mask = net_mask_flow_list[1]; delta = net_mask_flow_list[2];
                // xxxx_debug

                // net = ggml_cont(ctx, net);
                // ggml_set_name(net, "net");
                // ggml_set_output(net);

                // up_mask = ggml_cont(ctx, up_mask);
                // ggml_set_name(up_mask, "up_mask");
                // ggml_set_output(up_mask);

                // delta = ggml_cont(ctx, delta);
                // ggml_set_name(delta, "delta");
                // ggml_set_output(delta);


                // tensor [corr] size: [1, 324, 55, 128], min: -5.405947, max: 25.431339, mean: 0.233413
                // tensor [net] size: [1, 128, 55, 128], min: -1.0, max: 1.0, mean: 0.004631
                // tensor [up_mask] size: [1, 576, 55, 128], min: -18.799122, max: 9.778274, mean: -0.933968
                // tensor [delta] size: [1, 2, 55, 128], min: -1.231164, max: 1.291382, mean: -0.110575

                delta = ggml_cont(ctx, delta);
                coords1 = ggml_add(ctx, coords1, delta);
                flow = ggml_sub(ctx, coords1, coords0);
            }
        }

        // xxxx_debug
        {
            flow = ggml_cont(ctx, flow);
            ggml_set_name(flow, "m");
            ggml_set_output(flow);

            mask = ggml_cont(ctx, mask);
            ggml_set_name(mask, "up_mask");
            ggml_set_output(mask);

            // Info: ********************** m Tensor: 1x2x55x128
            // min: -186.5938, max: 190.5045, mean: -0.0859
            // Info: ********************** mask Tensor: 1x576x55x128
            // min: -18.0143, max: 8.6513, mean: -0.7198

            // tensor [m] size: [1, 2, 55, 128], min: -1.239185, max: 1.018547, mean: -0.111478
            // tensor [mask] size: [1, 576, 55, 128], min: -19.520226, max: 9.20424, mean: -0.050686
        }

        ggml_tensor_t *final_flow = upsample_flow(ctx, flow, mask);

        // Info: -------------- output_tensor Tensor: 1x2x440x1024
        // min: -178.7596, max: 178.8397, mean: 0.1127


        // tensor [final_flow] size: [1, 2, 440, 1024], min: -9.303978, max: 7.568922, mean: -0.884092
    	return final_flow;
    }
};

struct VideoFlowNetwork {
    RAFT net;
    GGMLModel model;

    int init(int device) {
        // -----------------------------------------------------------------------------------------
        net.set_device(device);
        net.start_engine();
        // net.dump();

        check_point(model.preload("models/video_flow_f32.gguf") == RET_OK);

        return RET_OK;
    }

    int load() {
        return net.load_weight(&model, "");
    }

    TENSOR *forward(TENSOR *input1_tensor, TENSOR *input2_tensor) {
        TENSOR *argv[2];
        argv[0] = input1_tensor ;
        argv[1] = input2_tensor ;

        load();
        return net.engine_forward(ARRAY_SIZE(argv), argv);
    }

    void exit() {
        model.clear();
        net.stop_engine();
    }
};

#endif // __RAFT__H__
