import argparse

import torch
import models
#from utils.google_utils import attempt_download

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov4-csp.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--cfg', type=str, default='cfg/yolov4.cfg', help='*.cfg path')
    parser.add_argument('--channels', type=int, default=3, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)

    # Input
    img = torch.zeros((opt.batch_size, opt.channels, *opt.img_size))  # image size(1,3,320,192) iDetection

    # Load PyTorch model
    #attempt_download(opt.weights)
    models.ONNX_EXPORT = True
    model = models.Darknet(opt.cfg)
    model.load_state_dict(torch.load(opt.weights, map_location=torch.device('cpu'))['model'])
    model.eval()
    #model = #.float()
    #print(model)
    #model.eval()
    model.export = True  # set Detect() layer export=True
    y = model(img)  # dry run

   
    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        model.fuse()  # only for ONNX
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)


    # Finish
    print('\nExport complete. Visualize with https://github.com/lutzroeder/netron.')
