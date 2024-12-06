#ifndef __RAFT__H__
#define __RAFT__H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"

/*
 FlowHead(
  (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(256, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu): ReLU()
) */

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

/*
 SepConvGRU(
  (convz1): Conv2d(384, 128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
  (convr1): Conv2d(384, 128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
  (convq1): Conv2d(384, 128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
  (convz2): Conv2d(384, 128, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
  (convr2): Conv2d(384, 128, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
  (convq2): Conv2d(384, 128, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
) */

struct SepConvGRU {
    // network hparams
    
    // network params
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
        // # horizontal
        // hx = torch.cat([h, x], dim=1)
        // z = torch.sigmoid(self.convz1(hx))
        // r = torch.sigmoid(self.convr1(hx))
        // q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        // h = (1 - z) * h + z * q

        // # vertical
        // hx = torch.cat([h, x], dim=1)
        // z = torch.sigmoid(self.convz2(hx))
        // r = torch.sigmoid(self.convr2(hx))
        // q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        // h = (1 - z) * h + z * q
        // # tensor [h] size: [1, 128, 55, 128], min: -1.0, max: 1.0, mean: 0.046477

        // return h

        ggml_tensor_t *hx, *z, *r, *rh, *q, *one_z;

        // # horizontal
         hx = ggml_concat(ctx, h, x, 2 /*dim on channel*/);
        z = ggml_sigmoid(ctx, convz1.forward(ctx, hx));
        one_z = ggml_dup(ctx, z);
        one_z = ggml_constant(ctx, one_z, 1.0);
        one_z = ggml_sub(ctx, one_z, z);
        r = ggml_sigmoid(ctx, convr1.forward(ctx, hx));
        // q
        rh = ggml_mul(ctx, r, h);
        rh = ggml_concat(ctx, rh, x, 2 /*dim on channel*/);
        q = ggml_tanh(ctx, convq1.forward(ctx, rh));
        h = ggml_add(ctx, ggml_mul(ctx, one_z, h), ggml_mul(ctx, z, q));

        // # vertical
        hx = ggml_concat(ctx, h, x, 2 /*dim on channel*/);
        z = ggml_sigmoid(ctx, convz2.forward(ctx, hx));
        one_z = ggml_constant(ctx, one_z, 1.0);
        one_z = ggml_sub(ctx, one_z, z);
        r = ggml_sigmoid(ctx, convr2.forward(ctx, hx));
        // q
        rh = ggml_mul(ctx, r, h);
        rh = ggml_concat(ctx, rh, x, 2 /*dim on channel*/);
        q = ggml_tanh(ctx, convq2.forward(ctx, rh));
        h = ggml_add(ctx, ggml_mul(ctx, one_z, h), ggml_mul(ctx, z, q));

    	return h;
    }
};

/*
 MotionEncoder(
  (convc1): Conv2d(324, 256, kernel_size=(1, 1), stride=(1, 1))
  (convc2): Conv2d(256, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (convf1): Conv2d(2, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
  (convf2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv): Conv2d(256, 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
) */

struct MotionEncoder {
    // network params
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

/*
 BasicUpdateBlock(
  (encoder): MotionEncoder(
    (convc1): Conv2d(324, 256, kernel_size=(1, 1), stride=(1, 1))
    (convc2): Conv2d(256, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (convf1): Conv2d(2, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
    (convf2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv): Conv2d(256, 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (gru): SepConvGRU(
    (convz1): Conv2d(384, 128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
    (convr1): Conv2d(384, 128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
    (convq1): Conv2d(384, 128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
    (convz2): Conv2d(384, 128, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
    (convr2): Conv2d(384, 128, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
    (convq2): Conv2d(384, 128, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
  )
  (flow_head): FlowHead(
    (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2): Conv2d(256, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (relu): ReLU()
  )
  (mask): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(256, 576, kernel_size=(1, 1), stride=(1, 1))
  )
) */

struct BasicUpdateBlock {
    // network params
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
    //     delta_flow = self.flow_head(net)

    //     # scale mask to balence gradients
    //     mask = 0.25 * self.mask(net)
    //     return net, mask, delta_flow
    std::vector<ggml_tensor_t *> forward(struct ggml_context* ctx, ggml_tensor_t* net, ggml_tensor_t* inp,
        ggml_tensor_t* corr, ggml_tensor_t* flow) {
        // tensor [net] size: [1, 128, 55, 128], min: -1.0, max: 1.0, mean: -0.000222
        // tensor [inp] size: [1, 128, 55, 128], min: 0.0, max: 5.488075, mean: 0.038178
        // tensor [corr] size: [1, 324, 55, 128], min: -5.405947, max: 25.431339, mean: 0.233413
        // tensor [flow] size: [1, 2, 55, 128], min: 0.0, max: 0.0, mean: 0.0

        std::vector<ggml_tensor_t *> xlist;

        // flow = ggml_cont(ctx, flow);
        // ggml_set_name(flow, "flow");
        // ggml_set_output(flow);

        corr = ggml_cont(ctx, corr);
        ggml_set_name(corr, "corr");
        ggml_set_output(corr);


        ggml_tensor_t *motion_feat = encoder.forward(ctx, flow, corr);

        motion_feat = ggml_cont(ctx, motion_feat);
        ggml_set_name(motion_feat, "motion_feat");
        ggml_set_output(motion_feat);


        // Info: ********************** flow Tensor: 1x2x55x128
        // min: -690.0397, max: 727.3568, mean: -2.0973
        // Info: ********************** corr Tensor: 1x324x55x128
        // min: -3.6457, max: 12.8752, mean: 0.0917
        // Info: ********************** motion_feat Tensor: 1x128x55x128
        // min: -690.0397, max: 727.3568, mean: 3.2731



        // tensor [motion_feat] size: [1, 128, 55, 128], min: 0.0, max: 27.771051, mean: 0.248971
        // tensor [motion_feat] size: [1, 128, 55, 128], min: -1.231163, max: 6.667092, mean: 0.124499
        // tensor [motion_feat] size: [1, 128, 55, 128], min: -1.213821, max: 6.57347, mean: 0.122071
        // tensor [motion_feat] size: [1, 128, 55, 128], min: -1.196342, max: 8.667865, mean: 0.124009
        // tensor [motion_feat] size: [1, 128, 55, 128], min: -1.202042, max: 6.649434, mean: 0.124851
        // tensor [motion_feat] size: [1, 128, 55, 128], min: -1.202976, max: 6.519124, mean: 0.125014
        // tensor [motion_feat] size: [1, 128, 55, 128], min: -1.212376, max: 6.62364, mean: 0.125172
        // tensor [motion_feat] size: [1, 128, 55, 128], min: -1.224716, max: 6.26437, mean: 0.125271
        // tensor [motion_feat] size: [1, 128, 55, 128], min: -1.247585, max: 6.365556, mean: 0.125302
        // tensor [motion_feat] size: [1, 128, 55, 128], min: -1.254894, max: 6.342296, mean: 0.125405
        // tensor [motion_feat] size: [1, 128, 55, 128], min: -1.249577, max: 6.391676, mean: 0.125365
        // tensor [motion_feat] size: [1, 128, 55, 128], min: -1.245121, max: 6.396429, mean: 0.125442
        // tensor [motion_feat] size: [1, 128, 55, 128], min: -1.240402, max: 6.534332, mean: 0.125424
        // tensor [motion_feat] size: [1, 128, 55, 128], min: -1.240623, max: 6.446226, mean: 0.125442
        // tensor [motion_feat] size: [1, 128, 55, 128], min: -1.232243, max: 6.580369, mean: 0.1254
        // tensor [motion_feat] size: [1, 128, 55, 128], min: -1.232037, max: 6.420638, mean: 0.125458
        // tensor [motion_feat] size: [1, 128, 55, 128], min: -1.230373, max: 6.605807, mean: 0.125387
        // tensor [motion_feat] size: [1, 128, 55, 128], min: -1.22847, max: 6.440721, mean: 0.125446
        // tensor [motion_feat] size: [1, 128, 55, 128], min: -1.227055, max: 6.571367, mean: 0.125326
        // tensor [motion_feat] size: [1, 128, 55, 128], min: -1.224991, max: 6.451536, mean: 0.125444



        inp = ggml_concat(ctx, inp, motion_feat, 2 /*dim on channe*/);
        net = gru.forward(ctx, net, inp);

        net = ggml_cont(ctx, net);
        ggml_set_name(net, "gru");
        ggml_set_output(net);


        ggml_tensor_t *delta_flow = flow_head.forward(ctx, net);

        delta_flow = ggml_cont(ctx, delta_flow);
        ggml_set_name(delta_flow, "flow_head");
        ggml_set_output(delta_flow);


        // Info: ********************** motion_feat Tensor: 1x128x55x128
        // min: -690.0397, max: 727.3568, mean: 3.2731
        // Info: ********************** gru Tensor: 1x128x55x128
        // min: -20.5265, max: 20.8681, mean: -0.0055
        // Info: ********************** flow_head Tensor: 1x2x55x128
        // min: -94.1873, max: 91.4377, mean: -4.3792



        ggml_tensor_t *mask;
        mask = mask_0.forward(ctx, net);
        mask = ggml_relu(ctx, mask);
        mask = mask_2.forward(ctx, mask);
        mask = ggml_scale(ctx, mask, 0.25f);

        xlist.push_back(net);
        xlist.push_back(mask);
        xlist.push_back(delta_flow);

    	return xlist;
    }
};

/*
 ResidualBlock(
  (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu): ReLU(inplace=True)
  (norm1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
  (norm2): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
  (downsample): Identity()
) */

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

        // x = ggml_nn_arange(ctx, x);

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
        x = ggml_relu(ctx,  ggml_add(ctx, x, y));

        // Info: ********************** InstanceResidualBlock Tensor: 2x128x55x128
        // min: 0.0000, max: 11.3031, mean: 0.7551

        // tensor [ResidualBlock] size: [2, 128, 55, 128], min: 1e-06, max: 11.252819, mean: 0.755113

        return x;
    }
};


/*
 BasicEncoder(
  (norm1): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
  (relu1): ReLU(inplace=True)
  (layer1): Sequential(
    (0): BatchResidualBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
      (norm1): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (norm2): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (downsample): Identity()
    )
    (1): BatchResidualBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
      (norm1): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (norm2): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (downsample): Identity()
    )
  )
  (layer2): Sequential(
    (0): BatchResidualBlock(
      (conv1): Conv2d(64, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (conv2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
      (norm1): InstanceNorm2d(96, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (norm2): InstanceNorm2d(96, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (norm3): InstanceNorm2d(96, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (downsample): Sequential(
        (0): Conv2d(64, 96, kernel_size=(1, 1), stride=(2, 2))
        (1): InstanceNorm2d(96, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (1): BatchResidualBlock(
      (conv1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
      (norm1): InstanceNorm2d(96, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (norm2): InstanceNorm2d(96, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (downsample): Identity()
    )
  )
  (layer3): Sequential(
    (0): BatchResidualBlock(
      (conv1): Conv2d(96, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
      (norm1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (norm2): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (norm3): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (downsample): Sequential(
        (0): Conv2d(96, 128, kernel_size=(1, 1), stride=(2, 2))
        (1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
    (1): BatchResidualBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
      (norm1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (norm2): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      (downsample): Identity()
    )
  )
  (conv2): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
) */

// BasicEncoder
struct BatchBasicEncoder {
    // network params
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

        // layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        // layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        // self.layer1 = self._make_layer(64, stride=1)

        layer1_0.in_planes = 64;
        layer1_0.planes = 64;
        layer1_0.stride = 1;
        layer1_0.create_weight_tensors(ctx);

        layer1_1.in_planes = 64;
        layer1_1.planes = 64;
        layer1_1.stride = 1;
        layer1_1.create_weight_tensors(ctx);


        // ResidualBlock(
        //   (conv1): Conv2d(64, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        //   (conv2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        //   (relu): ReLU(inplace=True)
        //   (norm1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        //   (norm2): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        //   (norm3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        //   (downsample): Sequential(
        //     (0): Conv2d(64, 96, kernel_size=(1, 1), stride=(2, 2))
        //     (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        //   )
        // )
        // ResidualBlock(
        //   (conv1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        //   (conv2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        //   (relu): ReLU(inplace=True)
        //   (norm1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        //   (norm2): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        //   (downsample): Identity()
        // )

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

        // x = ggml_nn_arange(ctx, x);
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

        // x = ggml_cont(ctx, x);
        // ggml_set_name(x, "BatchBasicEncoder");
        // ggml_set_output(x);

    	return x;
    }
};

// BasicEncoder
struct InstanceBasicEncoder {
    // network params
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

        // layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        // layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        // self.layer1 = self._make_layer(64, stride=1)

        // _make_layer:
        // 64 64 batch 1
        // 64 64 batch 1
        // --------------------------------------------------------------------------------
        layer1_0.in_planes = 64;
        layer1_0.planes = 64;
        layer1_0.stride = 1;
        layer1_0.create_weight_tensors(ctx);

        layer1_1.in_planes = 64;
        layer1_1.planes = 64;
        layer1_1.stride = 1;
        layer1_1.create_weight_tensors(ctx);

        // _make_layer:
        // 64 96 batch 2
        // 96 96 batch 2
        // --------------------------------------------------------------------------------
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
        // _make_layer:
        // 96 128 batch 2
        // 128 128 batch 2
        // --------------------------------------------------------------------------------
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
        // x = ggml_cont(ctx, x);
        // ggml_set_name(x, "x1");
        // ggml_set_output(x);

        x = conv1.forward(ctx, x);

        // x = ggml_cont(ctx, x);
        // ggml_set_name(x, "x2");
        // ggml_set_output(x);

        x = norm1.forward(ctx, x);

        // x = ggml_cont(ctx, x);
        // ggml_set_name(x, "x3");
        // ggml_set_output(x);


        x = ggml_relu(ctx, x);

        x = layer1_0.forward(ctx, x);
        x = layer1_1.forward(ctx, x);

        // x = ggml_cont(ctx, x);
        // ggml_set_name(x, "x4");
        // ggml_set_output(x);

        x = layer2_0.forward(ctx, x);
        x = layer2_1.forward(ctx, x);

        // x = ggml_cont(ctx, x);
        // ggml_set_name(x, "x5");
        // ggml_set_output(x);

        x = layer3_0.forward(ctx, x);
        x = layer3_1.forward(ctx, x);

        // x = ggml_cont(ctx, x);
        // ggml_set_name(x, "x6");
        // ggml_set_output(x);

        x = conv2.forward(ctx, x);

        // x = ggml_cont(ctx, x);
        // ggml_set_name(x, "x7");
        // ggml_set_output(x);



        // Info: ********************** images Tensor: 2x3x440x1024
        // min: -1.0000, max: 1.0000, mean: -0.2467

        // Info: ********************** x1 Tensor: 2x3x440x1024
        // min: -1.0000, max: 1.0000, mean: -0.2467
        // Info: ********************** x2 Tensor: 2x64x220x512
        // min: -7.5185, max: 7.8995, mean: 0.0257
        // Info: ********************** x3 Tensor: 2x64x220x512
        // min: -13.4140, max: 12.1434, mean: 0.0000

        // Info: ********************** x4 Tensor: 2x64x220x512
        // min: 0.0000, max: 25.7299, mean: 1.0922

        // Info: ********************** x5 Tensor: 2x96x110x256
        // min: 0.0000, max: 14.1641, mean: 1.0558

        // Info: ********************** x6 Tensor: 2x128x55x128
        // min: 0.0000, max: 11.4655, mean: 1.1087

        // Info: ********************** x7 Tensor: 2x256x55x128
        // min: -4.4848, max: 4.7935, mean: 0.0030


        // tensor [x1] size: [2, 3, 440, 1024], min: -1.0, max: 1.0, mean: -0.245923
        // tensor [x2] size: [2, 64, 220, 512], min: -7.565556, max: 7.899763, mean: 0.025623
        // tensor [x3] size: [2, 64, 220, 512], min: -13.541551, max: 13.151362, mean: 0.0
        // tensor [x4] size: [2, 64, 220, 512], min: 0.0, max: 25.720057, mean: 1.092221
        // tensor [x5] size: [2, 96, 110, 256], min: 0.0, max: 14.151134, mean: 1.055869
        // tensor [x6] size: [2, 128, 55, 128], min: 0.0, max: 11.478568, mean: 1.108787
        // tensor [x7] size: [2, 256, 55, 128], min: -4.488491, max: 5.076661, mean: 0.003168
        return x;
    }
};


/*
 RAFT(
  (cnet): BasicEncoder(
    (norm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    (relu1): ReLU(inplace=True)
    (layer1): Sequential(
      (0): BatchResidualBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu): ReLU(inplace=True)
        (norm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Identity()
      )
      (1): BatchResidualBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu): ReLU(inplace=True)
        (norm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Identity()
      )
    )
    (layer2): Sequential(
      (0): BatchResidualBlock(
        (conv1): Conv2d(64, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (conv2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu): ReLU(inplace=True)
        (norm1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm2): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(64, 96, kernel_size=(1, 1), stride=(2, 2))
          (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BatchResidualBlock(
        (conv1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu): ReLU(inplace=True)
        (norm1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm2): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Identity()
      )
    )
    (layer3): Sequential(
      (0): BatchResidualBlock(
        (conv1): Conv2d(96, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu): ReLU(inplace=True)
        (norm1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(96, 128, kernel_size=(1, 1), stride=(2, 2))
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BatchResidualBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu): ReLU(inplace=True)
        (norm1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Identity()
      )
    )
    (conv2): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
  )
  (fnet): BasicEncoder(
    (norm1): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    (relu1): ReLU(inplace=True)
    (layer1): Sequential(
      (0): BatchResidualBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu): ReLU(inplace=True)
        (norm1): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (norm2): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (downsample): Identity()
      )
      (1): BatchResidualBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu): ReLU(inplace=True)
        (norm1): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (norm2): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (downsample): Identity()
      )
    )
    (layer2): Sequential(
      (0): BatchResidualBlock(
        (conv1): Conv2d(64, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (conv2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu): ReLU(inplace=True)
        (norm1): InstanceNorm2d(96, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (norm2): InstanceNorm2d(96, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (norm3): InstanceNorm2d(96, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (downsample): Sequential(
          (0): Conv2d(64, 96, kernel_size=(1, 1), stride=(2, 2))
          (1): InstanceNorm2d(96, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        )
      )
      (1): BatchResidualBlock(
        (conv1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu): ReLU(inplace=True)
        (norm1): InstanceNorm2d(96, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (norm2): InstanceNorm2d(96, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (downsample): Identity()
      )
    )
    (layer3): Sequential(
      (0): BatchResidualBlock(
        (conv1): Conv2d(96, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu): ReLU(inplace=True)
        (norm1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (norm2): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (norm3): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (downsample): Sequential(
          (0): Conv2d(96, 128, kernel_size=(1, 1), stride=(2, 2))
          (1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        )
      )
      (1): BatchResidualBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (relu): ReLU(inplace=True)
        (norm1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (norm2): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (downsample): Identity()
      )
    )
    (conv2): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
  )
  (update_block): BasicUpdateBlock(
    (encoder): MotionEncoder(
      (convc1): Conv2d(324, 256, kernel_size=(1, 1), stride=(1, 1))
      (convc2): Conv2d(256, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (convf1): Conv2d(2, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
      (convf2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv): Conv2d(256, 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (gru): SepConvGRU(
      (convz1): Conv2d(384, 128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
      (convr1): Conv2d(384, 128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
      (convq1): Conv2d(384, 128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
      (convz2): Conv2d(384, 128, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
      (convr2): Conv2d(384, 128, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
      (convq2): Conv2d(384, 128, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
    )
    (flow_head): FlowHead(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(256, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU()
    )
    (mask): Sequential(
      (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): Conv2d(256, 576, kernel_size=(1, 1), stride=(1, 1))
    )
  )
) */

struct RAFT : GGMLNetwork {
    // network hparams
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
        // int C = (int)x->ne[2];
        // int B = (int)x->ne[3];

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

    // def bilinear_sampler(img, coords):
    //     # img.size() -- [7040, 1, 55, 128]
    //     # coords.size() -- [7040, 9, 9, 2]
    //     B, C, H, W = img.size()
    //     xgrid, ygrid = coords.split([1, 1], dim=3)

    //     xgrid = 2.0 * xgrid / (W - 1.0) - 1.0
    //     ygrid = 2.0 * ygrid / (H - 1.0) - 1.0

    //     # xgrid.size() -- [7040, 9, 9, 1]
    //     # ygrid.size() -- [7040, 9, 9, 1]
    //     grid = torch.cat([xgrid, ygrid], dim=3)
    //     img = F.grid_sample(img, grid, align_corners=True)

    //     return img # img.size() -- [7040, 1, 9, 9]

    ggml_tensor_t* bilinear_sampler(struct ggml_context* ctx, ggml_tensor_t* image, ggml_tensor_t* coords) {
        int W = (int)image->ne[0];
        int H = (int)image->ne[1];
        ggml_tensor_t * mesh_x = ggml_nn_slice(ctx, coords, 0 /*dim*/, 0 /*start*/, 1 /*stop*/, 1/*step*/);
        ggml_tensor_t * mesh_y = ggml_nn_slice(ctx, coords, 0 /*dim*/, 1 /*start*/, 2 /*stop*/, 1/*step*/);
        mesh_x = ggml_scale(ctx, mesh_x, 2.0f/(W - 1.0f));
        mesh_x = ggml_add_constant(ctx, mesh_x, -1.0);
        // mesh_x    f32 [1, 9, 9, 7040],  (view)

        mesh_y = ggml_scale(ctx, mesh_y, 2.0f/(H - 1.0f));
        mesh_y = ggml_add_constant(ctx, mesh_y, -1.0);
        // mesh_y    f32 [1, 9, 9, 7040],  (view)

        ggml_tensor_t *grid = ggml_concat(ctx, mesh_x, mesh_y, 0/*dim*/);
        // grid    f32 [2, 9, 9, 7040], 
        image = ggml_grid_sample(ctx, image, grid);
        // image    f32 [9, 9, 1, 7040], 

        return image; // f32 [9, 9, 1, 7040]
    }

    // def create_corr_pyramid(fmap1, fmap2) -> List[torch.Tensor]:
    //     # corr_levels: int, corr_radius: int === 4, 4
    //     B, C, H, W = fmap1.shape  # (1, 256, 55, 128)
    //     fmap1 = fmap1.view(B, C, H * W)
    //     fmap2 = fmap2.view(B, C, H * W)

    //     # C === 256 ==> torch.sqrt(torch.tensor(C).float()) === 16.0
    //     # corr = torch.matmul(fmap1.transpose(1, 2), fmap2) / torch.sqrt(torch.tensor(C).float())
    //     corr = torch.matmul(fmap1.transpose(1, 2), fmap2) / 16.0
    //     # size() -- [1, 7040, 7040]
    //     corr = corr.reshape(H * W, 1, H, W)  # ==> size() -- [7040, 1, 55, 128]

    //     corr_pyramid = []
    //     corr_pyramid.append(corr)
    //     corr_levels = 4
    //     for i in range(corr_levels - 1):
    //         corr = F.avg_pool2d(corr, 2, stride=2)
    //         corr_pyramid.append(corr)

    //     # corr_pyramid is list: len = 4
    //     #     tensor [item] size: [7040, 1, 55, 128], min: -7.369149, max: 25.431339, mean: 0.033188
    //     #     tensor [item] size: [7040, 1, 27, 64], min: -3.66336, max: 9.582128, mean: 0.032375
    //     #     tensor [item] size: [7040, 1, 13, 32], min: -2.107447, max: 4.198452, mean: 0.03262
    //     #     tensor [item] size: [7040, 1, 6, 16], min: -1.357178, max: 2.21133, mean: 0.03297
    //     return corr_pyramid


    std::vector<ggml_tensor_t *> create_corr_pyramid(struct ggml_context* ctx, ggml_tensor_t* fmap1, ggml_tensor_t* fmap2) {
        std::vector<ggml_tensor_t *> xlist;

        int W = (int)fmap1->ne[0];
        int H = (int)fmap1->ne[1];
        int C = (int)fmap1->ne[2];
        int B = (int)fmap1->ne[3];

        fmap1 = ggml_reshape_3d(ctx, fmap1, W*H, C, B);
        fmap2 = ggml_reshape_3d(ctx, fmap2, W*H, C, B);

        fmap1 = ggml_transpose(ctx, fmap1);
        ggml_tensor_t *corr = ggml_nn_mul_mat(ctx, fmap1, fmap2);
        corr = ggml_reshape_4d(ctx, corr, W, H, 1, H*W);
        corr = ggml_scale(ctx, corr, 1.0f/16.0);

        xlist.push_back(corr);

        // return xlist;

        for (int i = 0; i < 3; i++) {
            corr = ggml_pool_2d(ctx, corr, GGML_OP_POOL_AVG, 
                    2 /*kernel*/, 2/*kernel*/, 2 /*stride*/, 2/*stride*/, 0.0/*(float)padding*/, 0.0/*(float)padding*/);
            xlist.push_back(corr);
        }

        return xlist;
    }


    // def index_corr_volume(coords, corr_pyramid: List[torch.Tensor], mesh_grid_9x9):
    //     # tensor [coords] size: [1, 2, 55, 128], min: 0.0, max: 127.0, mean: 45.25
    //     coords = coords.permute(0, 2, 3, 1) # [1, 2, 55, 128] --> [1, 55, 128, 2]
    //     # tensor [coords] size: [1, 55, 128, 2], min: 0.0, max: 127.0, mean: 45.25

    //     B, H, W, N = coords.size()

    //     out_pyramid = []
    //     corr_levels = 4
    //     for i in range(corr_levels): # 4 
    //         centroid = coords.reshape(B*H*W, 1, 1, 2) / 2**i
    //         corr = bilinear_sampler(corr_pyramid[i], centroid + mesh_grid_9x9).view(B, H, W, -1)
    //         out_pyramid.append(corr)

    //     out = torch.cat(out_pyramid, dim=3)  # [1, 55, 128, 324]
    //     return out.permute(0, 3, 1, 2)  # [1, 324, 55, 128]
    ggml_tensor_t * index_corr_volume(struct ggml_context* ctx, ggml_tensor_t *coords, std::vector<ggml_tensor_t *>xlist) {
        ggml_tensor_t *centroid, *corr, *out_pyramid[4];

        int W = (int)coords->ne[0];
        int H = (int)coords->ne[1];
        int D = (int)coords->ne[2];
        int B = (int)coords->ne[3];

        // coords    f32 [1024, 440, 2, 1], 
        coords = ggml_cont(ctx, ggml_permute(ctx, coords, 1, 2, 0, 3)); // [W, H, 2, B] --> [2, W, H, B]
        // coords    f32 [2, 1024, 440, 1],  (permuted) (cont)

        float fscale = 1.0f;
        for (int i = 0; i < 4; i++) {
            // coords    f32 [2, 128, 55, 1],  (permuted) (cont)
            centroid = ggml_reshape_4d(ctx, coords, 2, 1, 1, B*H*W);
            centroid = ggml_repeat_ext(ctx, centroid, 1, 9, 9, 1);
            // centroid    f32 f32 [2, 9, 9, 7040],  (permuted) (cont) (reshaped)
            centroid = ggml_scale(ctx, centroid, fscale); fscale /= 2.0f;
            // mesh_grid_9x9    f32 [1, 9, 9, 2], mesh_grid_9x9

            // # centroid -- [7040, 1, 1, 2]
            // # mesh_grid_9x9 -- [9, 9, 2]
            // # (centroid + mesh_grid_9x9) -- [7040, 9, 9, 2]
            centroid = ggml_add(ctx, centroid, mesh_grid_9x9);

            corr = bilinear_sampler(ctx, xlist[i], centroid);
            // corr  -- f32 [9, 9, 1, 7040]

            corr = ggml_reshape_4d(ctx, corr, -1, W, H, B);
            // f32 [81, 128, 55, 1]
            out_pyramid[i] = corr;
        }

        ggml_tensor_t *out = ggml_cat(ctx, 4, out_pyramid[0], out_pyramid[1], out_pyramid[2], out_pyramid[3], 0/*dim*/);

        // [324, 128, 55, 1]
        out = ggml_cont(ctx, ggml_permute(ctx, out, 2, 0, 1, 3)); // [C, W, H, B] -> [W, H, C, B]
        // out    f32 [128, 55, 324, 1],  (permuted) (cont)

        return out;
    }

    // def upsample_flow(self, flow, mask):
    //     """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""

    //     # tensor [flow] size: [1, 2, 55, 128], min: -1.231163, max: 1.291382, mean: -0.110575
    //     # tensor [mask] size: [1, 576, 55, 128], min: -18.799122, max: 9.778274, mean: -0.933968
    //     N, _, H, W = flow.shape # [1, 2, 55, 128

    //     # ggml_debug
    //     # tensor [flow] size: [1, 2, 55, 128], min: -1.231163, max: 1.291382, mean: -0.110575
    //     up_flow = F.unfold(8 * flow, [3, 3], padding=1)
    //     # tensor [up_flow] size: [1, 18, 7040], min: -9.849304, max: 10.331055, mean: -0.872977

    //     # ggml_debug
    //     mask2 = mask.view(1, 9, 64, H*W)
    //     mask2 = torch.softmax(mask2, dim=1)
    //     up_flow2 = up_flow.view(2, 9, 1, H*W)
    //     up_flow2 = torch.sum(mask2 * up_flow2, dim=1)
    //     # ==> up_flow2 -- [2, 64, H*W] -> [2, 64, H, W] --> pixel_shuffle --> [2, 1, 8*H, 8*W] --> [1, 2, 8*H, 8*W]
    //     up_flow2 = F.pixel_shuffle(up_flow2.view(2, 64, H, W), 8)
    //     # up_flow2 = up_flow2.permute(1, 0, 2, 3)
    //     return up_flow2.reshape(N, 2, 8 * H, 8 * W)


    ggml_tensor_t* upsample_flow(ggml_context_t* ctx, ggml_tensor_t *flow, ggml_tensor_t *mask) {
        int B = (int)flow->ne[3];
        int D = (int)flow->ne[2];
        int H = (int)flow->ne[1];
        int W = (int)flow->ne[0];
        // flow    f32 [128, 55, 2, 1], 

        flow = ggml_scale(ctx, flow, 8.0f);
        flow = ggml_cont(ctx, flow);
        ggml_set_name(flow, "flow");
        ggml_set_output(flow);


        ggml_tensor_t *up_flow;
        up_flow = ggml_nn_unfold(ctx, flow, 3 /*k0*/, 3 /*k1*/,
                1 /*s0*/, 1/*s1*/, 1/*p0*/, 1/*p1*/, 1/*d0*/, 1/*d1*/);
        // [128*55, 18, 1]
        up_flow = ggml_cont(ctx, up_flow);
        ggml_set_name(up_flow, "up_flow");
        ggml_set_output(up_flow);



        up_flow = ggml_reshape_4d(ctx, up_flow, H*W, 1, 9, 2);
        mask = ggml_reshape_4d(ctx, mask, H*W, 64, 9, 1);
        mask = ggml_softmax(ctx, mask, 2 /*dim*/);


        // mask    f32 [7040, 64, 9, 1], 
        // up_flow    f32 [7040, 1, 9, 2],  (permuted) (cont) (reshaped) (reshaped)

        // # tensor [mask] size: [1, 9, 64, 7040], min: 0.0, max: 0.980355, mean: 0.111111
        // # tensor [up_flow] size: [2, 9, 1, 7040], min: -9.789093, max: 8.164978, mean: -0.868043
        mask = ggml_repeat_ext(ctx, mask, 1, 1, 1, 2);
        up_flow = ggml_repeat_ext(ctx, up_flow, 1, 64, 1, 1);
        up_flow = ggml_mul(ctx, mask, up_flow);

        // up_flow = ggml_cont(ctx, up_flow);
        // ggml_set_name(up_flow, "up_flow");
        // ggml_set_output(up_flow);



        // up_flow    f32 [7040, 64, 9, 2], ==> 7040, 64, 2

        // up_flow2 = torch.sum(mask2 * up_flow2, dim=1)
        // # ==> up_flow2 -- [2, 64, H*W] -> [2, 64, H, W] --> pixel_shuffle --> [2, 1, 8*H, 8*W] --> [1, 2, 8*H, 8*W]
 
        up_flow = ggml_mean_ext(ctx, up_flow, 2 /*dim*/);
        up_flow = ggml_scale(ctx, up_flow, 9.0f);
        // up_flow2    f32 f32 [7040, 64, 1, 2], 
        // # tensor [up_flow] size: [2, 64, 7040], min: -9.221816, max: 7.563138, mean: -0.876239

        up_flow = ggml_reshape_4d(ctx, up_flow, W, H, 64, 2);
        up_flow = ggml_shuffle(ctx, up_flow, 8);
        up_flow = ggml_reshape_4d(ctx, up_flow, 8*W, 8*H, 2, B);

        return up_flow;
    }

    ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[]) {
        std::vector<ggml_tensor_t *>xlist;
        ggml_tensor_t *image1, *image2, *up_mask, *m, *coords0, *coords1, *net, *inp;

        GGML_UNUSED(argc);
        image1 = argv[0];
        image2 = argv[1];

        mesh_grid_9x9 = ggml_cont(ctx, ggml_permute(ctx, mesh_grid_9x9, 3, 1, 2, 0)); // [2, 9, 9, 1] -> [1, 9, 9, 2]

        image1 = resize_pad(ctx, image1);
        image2 = resize_pad(ctx, image1);
        int W = (int)image1->ne[0];
        int H = (int)image1->ne[1];
        int C = (int)image1->ne[2];
        int B = (int)image1->ne[3];

        image1 = ggml_scale(ctx, image1, 2.0);
        image1 = ggml_add_constant(ctx, image1, -1.0);

        image2 = ggml_scale(ctx, image2, 2.0);
        image2 = ggml_add_constant(ctx, image2, -1.0);

        {
            ggml_tensor_t *images, *fmaps, *fmap1, *fmap2;
            images = ggml_concat(ctx, image1, image2, 3 /*dim on Batch*/);

            // images = ggml_cont(ctx, images);
            // ggml_set_name(images, "images");
            // ggml_set_output(images);

            fmaps = fnet.forward(ctx, images);
            fmap1 = ggml_nn_slice(ctx, fmaps, 3/*dim*/, 0/*start*/, 1/*stop*/, 1/*step*/);
            fmap2 = ggml_nn_slice(ctx, fmaps, 3/*dim*/, 1/*start*/, 2/*stop*/, 1/*step*/);

            // fmaps = ggml_cont(ctx, fmaps);
            // ggml_set_name(fmaps, "fmaps");
            // ggml_set_output(fmaps);


            // Info: ********************** images Tensor: 2x3x440x1024
            // min: -1.0000, max: 1.0000, mean: -0.2467
            // Info: ********************** fmaps Tensor: 2x256x55x128
            // min: -4.4848, max: 4.7935, mean: 0.0030

            // tensor [images] size: [2, 3, 440, 1024], min: -1.0, max: 1.0, mean: -0.245923
            // tensor [fmaps] size: [2, 256, 55, 128], min: -4.488491, max: 5.076661, mean: 0.003168


            xlist = create_corr_pyramid(ctx, fmap1, fmap2);

            // xlist[0] = ggml_cont(ctx, xlist[0]);
            // ggml_set_name(xlist[0], "xlist0");
            // ggml_set_output(xlist[0]);

            // xlist[1] = ggml_cont(ctx, xlist[1]);
            // ggml_set_name(xlist[1], "xlist1");
            // ggml_set_output(xlist[1]);

            // xlist[2] = ggml_cont(ctx, xlist[2]);
            // ggml_set_name(xlist[2], "xlist2");
            // ggml_set_output(xlist[2]);

            // xlist[3] = ggml_cont(ctx, xlist[3]);
            // ggml_set_name(xlist[3], "xlist3");
            // ggml_set_output(xlist[3]);

            // Info: ********************** xlist0 Tensor: 7040x1x55x128
            // min: -7.2816, max: 27.4016, mean: 0.0334
            // Info: ********************** xlist1 Tensor: 7040x1x27x64
            // min: -4.1553, max: 9.4860, mean: 0.0326
            // Info: ********************** xlist2 Tensor: 7040x1x13x32
            // min: -2.0840, max: 3.9671, mean: 0.0329
            // Info: ********************** xlist3 Tensor: 7040x1x6x16
            // min: -1.3456, max: 2.7473, mean: 0.0332

            // # corr_pyramid is list: len = 4
            // #     tensor [item] size: [7040, 1, 55, 128], min: -7.369149, max: 25.431339, mean: 0.033188
            // #     tensor [item] size: [7040, 1, 27, 64], min: -3.66336, max: 9.582128, mean: 0.032375
            // #     tensor [item] size: [7040, 1, 13, 32], min: -2.107447, max: 4.198452, mean: 0.03262
            // #     tensor [item] size: [7040, 1, 6, 16], min: -1.357178, max: 2.21133, mean: 0.03297
        }

        {
            ggml_tensor_t *cnet_temp = cnet.forward(ctx, image1);

            int N = (int)cnet_temp->ne[2]/2;
            net = ggml_nn_slice(ctx, cnet_temp, 2/*dim on channel*/, 0, N, 1/*step*/);
            inp = ggml_nn_slice(ctx, cnet_temp, 2/*dim on channel*/, N, 2*N, 1/*step*/);
            net = ggml_tanh(ctx, net);
            inp = ggml_relu(ctx, inp);

            // tensor [net] size: [1, 128, 55, 128], min: -1.0, max: 1.0, mean: -0.000222
            // tensor [inp] size: [1, 128, 55, 128], min: 0.0, max: 5.488075, mean: 0.038178

            // Info: ********************** net Tensor: 1x128x55x128
            // min: -1.0000, max: 1.0000, mean: -0.0002
            // Info: ********************** inp Tensor: 1x128x55x128
            // min: 0.0000, max: 5.4858, mean: 0.0382
        }

        coords0 = ggml_grid_mesh(ctx, B, H/8, W/8, 0/*norm*/);
        coords1 = coords0;
        // # tensor [coords0] size: [1, 2, 55, 128], min: 0.0, max: 127.0, mean: 45.25
        // --------------------------------------------------------------------------------------------------------
        {
            ggml_tensor_t *corr, *delta_flow;

            std::vector<ggml_tensor_t *>ylist;
            // xxxx_debug
            for (int i = 0; i < 1; i++) {
                corr = index_corr_volume(ctx, coords1, xlist);
                // # tensor [corr] size: [1, 324, 55, 128], min: -4.405475, max: 21.874269, mean: 0.16877
                m = ggml_sub(ctx, coords1, coords0);

                // tensor [net] size: [1, 128, 55, 128], min: -1.0, max: 1.0, mean: -0.000222
                // tensor [inp] size: [1, 128, 55, 128], min: 0.0, max: 5.488075, mean: 0.038178
                // tensor [corr] size: [1, 324, 55, 128], min: -5.405947, max: 25.431339, mean: 0.233413
                // tensor [m] size: [1, 2, 55, 128], min: 0.0, max: 0.0, mean: 0.0

                // net = ggml_nn_arange(ctx, net);
                // inp = ggml_nn_arange(ctx, inp);
                // corr = ggml_nn_arange(ctx, corr);
                // m = ggml_nn_arange(ctx, m);

                ylist = update_block.forward(ctx, net, inp, corr, m);

                net = ylist[0]; up_mask = ylist[1]; delta_flow = ylist[2];
                // net = ggml_cont(ctx, net);
                // ggml_set_name(net, "net");
                // ggml_set_output(net);

                // up_mask = ggml_cont(ctx, up_mask);
                // ggml_set_name(up_mask, "up_mask");
                // ggml_set_output(up_mask);

                // delta_flow = ggml_cont(ctx, delta_flow);
                // ggml_set_name(delta_flow, "delta_flow");
                // ggml_set_output(delta_flow);

                // Info: ********************** up_mask Tensor: 1x576x55x128
                // min: -4.3849, max: 2.1323, mean: -0.1609
                // Info: ********************** delta_flow Tensor: 1x2x55x128
                // min: -4.1365, max: 4.3125, mean: -1.0573

                // tensor [up_mask] size: [1, 576, 55, 128], min: -3.754837, max: 1.990122, mean: -0.048588
                // tensor [delta_flow] size: [1, 2, 55, 128], min: -3.181628, max: 1.919293, mean: 0.090983


                coords1 = ggml_add(ctx, coords1, delta_flow);
            }
        }
        m = ggml_sub(ctx, coords1, coords0);

        // tensor [m] size: [1, 2, 55, 128], min: 0.0, max: 0.0, mean: 0.0
        // tensor [up_mask] size: [1, 576, 55, 128], min: -19.438438, max: 9.087296, mean: -0.03894

        ggml_tensor_t *flow_up = upsample_flow(ctx, m, up_mask);
        flow_up = ggml_cont(ctx, flow_up);

        // tensor [flow_up] size: [1, 2, 440, 1024], min: -9.221816, max: 7.563138, mean: -0.876239

        // Info: -------------- output_tensor Tensor: 1x2x440x1024
        // min: -387.5398, max: 622.5826, mean: -3.3714

    	return flow_up;
    }
};

struct VideoFlowNetwork {
    RAFT net;
    GGMLModel model;

    int init(int device) {
        // -----------------------------------------------------------------------------------------
        net.set_device(device);
        net.start_engine();
        net.dump();

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
