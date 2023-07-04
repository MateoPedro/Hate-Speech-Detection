# Hate Speech Detection
Federated learning project aiming to detect hate speech in online community chats while respecting user's privacy

The file, hate_flower_final.ipynb, is a notebook that sets up a Flower client and runs a simulation of the server and multiple clients. To run this, upload the notebook to Google Colab and choose a premium GPU instance. Also, upload the noisy_test.csv and noisy_train.csv files to the Colab instance. Run all cells to install the dependencies and run the experiment. 

load_data() loads the data based on NOISY_DATA and POISON_TRAIN flags, which are settings to set which experiments to run. If both of those flags are false, the original data from hate_speech18 from Hugging Face is loaded.

train() and test() are the training and testing routines for the loaded model similar to standard PyTorch training and testing loops.

The pretrained alBERT model is Hugging Face’s AutoModelForSequenceClassification module.

HATEClient is a flower client class that defines the behavior for our hate speech model. This enables the federation of the model.

The weighted_average function provides a way to aggregate the metrics distributed among the clients.

The start_simulation function runs the simulation. These simulation runs will output the performance metrics.
