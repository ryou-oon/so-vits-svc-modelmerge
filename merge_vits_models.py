import torch
import argparse


def load_vits_model(model_path):
    checkpoint_dict = torch.load(model_path, map_location='cpu')
    return checkpoint_dict


def merge_models(model_a, model_b, checkpoint_path, alpha=0.5):
    merged_state_dict = {}
    state_dict = model_a['model']
    iteration = model_a['iteration']
    learning_rate = model_a['learning_rate']
    optimizer = model_a['optimizer']

    # Merge the two models using the specified alpha value
    for key in state_dict.keys():
        merged_state_dict[key] = alpha * model_a['model'][key] + (1 - alpha) * model_b['model'][key]

    # Save the merged model
    torch.save({'model': merged_state_dict,
                'iteration': iteration,
                'optimizer': optimizer,
                'learning_rate': learning_rate}, checkpoint_path)

def main(args):
    # Load the models
    model_a = load_vits_model(args.model_path_a)
    model_b = load_vits_model(args.model_path_b)

    # Merge the models
    merged_model = merge_models(model_a, model_b, args.output_path, alpha=args.alpha)

    print("Merge complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge two VITS models.')
    parser.add_argument('--model_path_a', type=str, required=True, help='Path to the first model file.')
    parser.add_argument('--model_path_b', type=str, required=True, help='Path to the second model file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the merged model.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for the first model (default: 0.5).')

    args = parser.parse_args()
    main(args)