import torch
import random
import numpy as np
import pandas as pd
import time
import datetime
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.distributions.categorical import Categorical

import sys

from optim.FishLeg import FishLeg, FISH_LIKELIHOODS

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
 
# Set the seed value all over the place to make this reproducible.
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

if __name__ == "__main__":
    epochs = 6
    eta_adam = 1e-4
    eta_fl = 0.01
    eta_aux = 1e-4
    beta = 0.9
    damping = 5e-1

    seed_val = 43
 
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
 
    # If there's a GPU available...
    if torch.cuda.is_available():    
 
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
 
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
 


    likelihood = FISH_LIKELIHOODS["bernoulli"](device=device)
 
    def nll(model, data):
        data_x, data_y = data
        token_ids, input_mask = data_x[:, 0], data_x[:, 1]
        loss, _ = model(token_ids,
            token_type_ids=None,
            attention_mask=input_mask,
            labels=data_y)
        return loss
    
    def draw(model, data):
        data_x, _ = data
        token_ids, input_mask = data_x[:, 0], data_x[:, 1]
        pred_y = model(token_ids,
            token_type_ids=None,
            attention_mask=input_mask)[0]
        draw_y = Categorical(logits=pred_y).sample()
        return (data_x, draw_y)
 

 
    # Load the dataset into a pandas dataframe.
    df = pd.read_csv("/scratches/cblgpu07/rx220/rx220/.cache/cola_public/raw/in_domain_train.tsv", 
                     delimiter='\t', 
                     header=None, 
                     names=['sentence_source', 'label', 'label_notes', 'sentence'])
 
    # Report the number of sentences.
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))
    
    # Get the lists of sentences and their labels.
    sentences = df.sentence.values
    labels = df.label.values
    

    
    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    # Print the original sentence.
    print(' Original: ', sentences[0])
    
    # Print the sentence split into tokens.
    print('Tokenized: ', tokenizer.tokenize(sentences[0]))
    
    # Print the sentence mapped to token ids.
    print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))
 
    max_len = 0
    
    # For every sentence...
    for sent in sentences:
    
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
    
        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))
    
    print('Max sentence length: ', max_len)
 
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []
 
    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 64,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
    
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
    
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
 
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)
    labels = torch.tensor(labels, device=device)
    
    # Print sentence 0, now as a list of IDs.
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])
 

 
    input_x = torch.stack((input_ids, attention_masks), dim=1)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_x, labels)
    # Create a 90-10 train-validation split.
    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
 

 
    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.  
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        return_dict = False
    )
 
    # Tell pytorch to run this model on the GPU.
    # model.cuda()

    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    
    print('The BERT model has {:} different named parameters.\n'.format(len(params)))
    
    print('==== Embedding Layer ====\n')
    
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    
    print('\n==== First Transformer ====\n')
    
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    
    print('\n==== Output Layer ====\n')
    
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    

 
    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 4
 
 
    # The DataLoader needs to know our batch size for training, so we specify it
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch
    # size of 16 or 32.
    batch_size = 16
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )
    
    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    aux_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

 
    optimizer = FishLeg(
        model,
        draw,
        nll,
        aux_dataloader,
        likelihood,
        fish_lr=eta_fl,
        damping=damping,
        weight_decay=1e-4,
        beta=beta,
        update_aux_every=10,
        aux_lr=eta_aux,
        aux_betas=(0.9, 0.999),
        aux_eps=1e-8,
        module_names=[
            "re:bert.encoder.layer.*.attention",
            "re:bert.encoder.layer.*.intermediate.dense",
            "re:bert.encoder.layer.*.output.dense"
        ],
        config=model.config,
        device=device,
    )

        

    
    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []
    # Measure the total training time for the whole run.
    total_t0 = time.time()
    
    # For each epoch...
    for epoch_i in range(0, epochs):
    
        # ========================================
        #               Training
        # ========================================
    
        # Perform one full pass over the training set.
    
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
    
        # Measure how long the training epoch takes.
        t0 = time.time()
    
        # Reset the total loss for this epoch.
        total_train_loss = 0
        total_samples = 0
        correct_samples = 0
        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()
    
        # For each batch of training data...
        for step, (inputs, b_labels) in enumerate(train_dataloader):
            inputs, b_labels = inputs.to(device), b_labels.to(device)
            optimizer.zero_grad()

            loss = nll(optimizer.model, [inputs, b_labels])
            loss.backward()
            optimizer.step()
    
            total_train_loss += loss.item()
        
            b_input_ids, b_input_mask = inputs[:, 0], inputs[:, 1]
        
    
            _, predicted = optimizer.model(b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask)[0].max(1)
            total_samples += b_labels.size(0)
            correct_samples += predicted.eq(b_labels).sum().item()
    
            
            if step % 10 == 0:
                print('Loss: {:.3f} | Acc: {:.3f}%% ({:d}/{:d})'.format(
                        total_train_loss/(step+1), 
                        100.*correct_samples/total_samples, 
                        correct_samples, total_samples)) 
    
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            
    
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
    
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
        
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
    
        print("")
        print("Running Validation...")
    
        t0 = time.time()
    
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()
    
        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        correct_eval_samples = 0
        total_eval_samples = 0
    
        # Evaluate data for one epoch        
        for step, (inputs, b_labels) in enumerate(validation_dataloader):
            # inputs, b_labels = inputs.to(device), b_labels.to(device)
    
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        
    
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                loss = nll(optimizer.model, [inputs, b_labels])
            
                b_input_ids, b_input_mask = inputs[:, 0], inputs[:, 1]
                _, predicted = optimizer.model(b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask)[0].max(1)
                total_eval_samples += b_labels.size(0)
                correct_eval_samples += predicted.eq(b_labels).sum().item()
            
            # Accumulate the validation loss.
            total_eval_loss += loss.item()
           
        # Report the final accuracy for this validation run.
        avg_val_accuracy = 100.*correct_eval_samples/total_eval_samples
        print("  Validation Accuracy : {0:.2f}".format(avg_val_accuracy))
    
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
    
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
    
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
    
        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
    
    print("")
    print("Training complete!")
    
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    
