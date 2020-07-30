import argparse
from style_transfer.solver import transfer_style

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--content-image-path', required=True, help="path to the content image.")
    parser.add_argument('--style-image-path', required=True, help="path to the style image.")
    parser.add_argument('--cuda', action='store_true', help="use cuda.")
    parser.add_argument('--image-size', type=int, default=512, help="size to resize to.")
    parser.add_argument('--epochs', type=int, default=26, help="number of epochs to train for.")
    parser.add_argument('--content-weight', type=float, default=1, help="weight for the content loss.")
    parser.add_argument('--style-weight', type=str, default=1000000, help="weights for the style loss (if list, comma separated).")
    parser.add_argument('--content-layers', type=str, default="conv_4_2", help="layers used to obtain content representation (comma separated).")
    parser.add_argument('--style-layers', type=str, default="conv_1_1,conv_2_1,conv_3_1,conv_4_1,conv_5_1", help="layers used to obtain style representation (comma separated).")
    parser.add_argument('--save-path', type=str, default=None, help="path to save the output image (as JPEG).")

    args = parser.parse_args();

    # parsing variable length arguments
    args.style_weight = list(map(float, args.style_weight.split(",")))
    if len(args.style_weight) == 1:
        args.style_weight = args.style_weight[0]
    args.content_layers = list(map(str.strip, args.content_layers.split(",")))
    args.style_layers = list(map(str.strip, args.style_layers.split(",")))
    
    # calling function
    transfer_style(**vars(args))