from dataset import createDatasets
from model import loadEncoder, NeuralNetwork, NeuralNetworkDemographics
from solver import full_evaluation
import torch
import sys

dementia_path = 'notebooks/data/dementia_clocks/images/' # path for clock drawings from patients with dementia
control_path = 'notebooks/data/control_clocks/images/' # path for clock drawings from patients marked as control

# Demographics files should include the filename, then age, race, gender, and years of education.
dementia_demo_path = "notebooks/data/dementia_demographics.csv" # path for demographics .csv file of patients with dementia
control_demo_path = "notebooks/data/control_demographics.csv" # path for demographics .csv file of patients marked as control
rfvae_path = "notebooks/10d_fvae" # path to pretrained RFVAE used for encoding the clock drawings 
vae_dims = 10 # number of dimensions in the VAE

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        useDemographics = True
    else:
        useDemographics = False

    encoder = loadEncoder(rfvae_path, vae_dims)
    train_x, train_y, test_x, test_y, train_demo, test_demo = \
        createDatasets(dementia_path, control_path, useDemographics, dementia_demo_path, control_demo_path)

    if useDemographics:
        model = NeuralNetworkDemographics(encoder).to(device)
    else:
        model = NeuralNetwork(encoder).to(device)
    full_evaluation(model, train_x, train_y, test_x, test_y, train_demo, test_demo)