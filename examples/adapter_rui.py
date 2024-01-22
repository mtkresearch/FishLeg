import torch
import time
import argparse
import numpy as np
from datasets import load_dataset
from functools import partial
from itertools import chain
from transformers import default_data_collator
from transformers import get_scheduler
from transformers import GPT2AdapterModel, GPT2TokenizerFast
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader
from optim.FishLeg import FishLeg, FISH_LIKELIHOODS
from optim.FishLeg.fishleg_likelihood import BernoulliLikelihood
from torch.optim import AdamW, SGD

def evaluate_perplexity(
        model, inputs, seq_len, max_length, device, 
        head_name='lm_head',
        stride=512
):
    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = inputs.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids, head=head_name)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl

def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def preprocess_dataset(dataset, tokenizer):
    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")
    # Encode the input data
    dataset = dataset.map(encode_batch, batched=True)
    # The transformers model expects the target class column to be named "labels"
    dataset = dataset.rename_column("label", "labels")
    # Transform to pytorch tensors and only output the required columns
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-lr', type=float, help='fishleg learning rate', default=0.05)
    argparser.add_argument('-auxlr', type=float, help='fishleg auxilary learning rate', default=1e-4)
    argparser.add_argument('-sgdlr', type=float, help='fishleg sgd learning rate', default=0.01)
    argparser.add_argument('-pretrain', type=int, help='fishleg number of pretraining', default=0)
    argparser.add_argument('-damping', type=float, help='fishleg damping', default=0.5)
    argparser.add_argument('-weightdecay', type=float, help='lambda for weight decay', default=1.)
    argparser.add_argument('-uae', type=int, help='update aux every x iteration', default=10)
    argparser.add_argument('-norm', action="store_true", help='normalize u or not', default=False)
    argparser.add_argument('-diff', action="store_true", help='whether to use difference of gradients', default=False)
    argparser.add_argument('-save_every', type=int, help='saving every iterations', default=100)
    argparser.add_argument('-print_every', type=int, help='printing every iterations', default=10)
    argparser.add_argument('-num_epochs', type=int, help='number of eopchs', default=5)
    argparser.add_argument('--workdir', type=str, help='directory to save result', default='result/')
    argparser.add_argument('--exp', type=str, help='optimizer', default='fish')    
    args = argparser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    result_file = args.workdir + 'result_hp_pretrain.txt'


    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    dataset = load_dataset("imdb")
    dataset = preprocess_dataset(dataset, tokenizer)

    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
    column_names = list(raw_datasets["train"].features)
    def tokenize_function(examples):
        output = tokenizer(examples['text'])
        return output

    tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    lm_datasets = tokenized_datasets.map(
                partial(group_texts, block_size=100),
                batched=True,
            )

    for trial in range(1):

        lr = args.lr #np.logspace(-3,-1.5, num=30)[np.random.randint(0,30)]
        auxlr = args.auxlr #np.logspace(-5,-3, num=30)[np.random.randint(0,30)]
        sgdlr = args.sgdlr #np.logspace(-3,-1.5, num=30)[np.random.randint(0,30)]
        damping = args.damping #np.logspace(-3,1, num=30)[np.random.randint(0,30)]
        pretrain = args.pretrain #[0, 10, 50, 100, 200, 500, 1000, 2000][np.random.randint(8)]
        weight_decay = args.weightdecay

        ## load pretrained model and tokenizer
        model = GPT2AdapterModel.from_pretrained('gpt2').to(device)
        model.resize_token_embeddings(len(tokenizer))

        head_name = 'lm_head'
        max_length = model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)
        perplexity = partial(evaluate_perplexity,
            inputs=encodings,
            seq_len=seq_len,
            max_length=max_length,
            device=device,
            stride=stride,
            head_name=head_name
            )

        para_name = 'adapter'
        model.add_causal_lm_head(head_name)

        model.eval()
        eval_perp = perplexity(model).detach().cpu().numpy()
        print('Initial perplexity on wiki test: {:.4f}'.format(eval_perp)) 
        model.train()

        model.add_adapter("adapter_movie")
        model.add_classification_head(
            "imdb",
            num_labels=2,
            id2label={ 0: "👎", 1: "👍"}
        )
        model.train_adapter("adapter_movie")

        data_collator = default_data_collator
        corpus_dataloader = DataLoader(
            lm_datasets['train'],
            batch_size=8,
            shuffle=True,
            collate_fn=data_collator,
        )   
        train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=32)
        aux_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=64)

        def nll_movie(model, inputs):
            outputs = model(**inputs, head='imdb')
            pred_y = outputs['logits']
            data_y = inputs['labels']
            return -Categorical(logits=pred_y).log_prob(data_y).mean()

        def nll(model, inputs):
            outputs = model(**inputs, head='lm_head')
            pred_y = outputs['logits'][..., :-1, :].contiguous()
            data_y = inputs['labels'][..., 1:].contiguous()
            return -Categorical(
                        logits=pred_y.view(-1, model.config.vocab_size)
                    ).log_prob(data_y.view(-1)).mean()

        def draw(model, inputs):
            outputs = model(**inputs, head='lm_head')
            pred_y = outputs['logits'][..., :-1, :].contiguous()
            prefix = pred_y.shape[:-1]
            samples = inputs.copy()
            sampled_label = Categorical(
                        logits=pred_y.view(-1,model.config.vocab_size)
                    ).sample()
            sampled_label = torch.concat(
                    [
                        torch.zeros((8,1), 
                            device=device, 
                            dtype=sampled_label.dtype), 
                        sampled_label.view(*prefix)
                    ], dim=-1)
            samples['labels'] = sampled_label
            return samples

        def evaluate(model, inputs):
            outputs = model(**inputs, head='imdb')

            preds = np.argmax(outputs['logits'].detach().cpu().numpy(), axis=1)
            acc =  (preds == inputs['labels'].cpu().numpy()).mean()
            nll = -Categorical(logits=outputs['logits']).log_prob(inputs['labels']).mean()

            return nll, acc

        if args.exp == 'fish':

            if args.uae == 0:
                num_steps = args.pretrain
            else:
                num_steps = int(args.pretrain + args.num_epochs * len(train_dataloader) / args.uae)
            optimizer = FishLeg(model, draw, nll, corpus_dataloader,
                  likelihood=FISH_LIKELIHOODS['bernoulli'](device=device),
                  fish_lr=lr,
                  aux_lr=auxlr,
                  damping=damping,
                  update_aux_every=args.uae,
                  weight_decay=weight_decay,
                  device=device,
                  num_steps=num_steps,
                  para_name=para_name,
                  batch_speedup=False,
                  fine_tune=True,
                  initialization='zero',
                  warmup=0, # use sgd
                  scale=np.sqrt(sgdlr/lr),
                  normalization=args.norm
            )

            free_param = [
                param
                for name, param in model.named_parameters()
                if para_name not in name and param.requires_grad
            ]
            print("Adam optimized Param: ", [
                    name for name, param in model.named_parameters()
                    if para_name not in name and param.requires_grad
                ])
            free_opt = AdamW(free_param,lr=5e-5,weight_decay=1e-2)
            num_training_steps = args.num_epochs * len(train_dataloader)
            free_scheduler = get_scheduler(
                    name='linear', optimizer=free_opt,
                    num_warmup_steps=0,
                    num_training_steps=num_training_steps
            )
        elif args.exp == 'adam':
            model.to(device)
            optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=args.weightdecay)
            num_training_steps = args.num_epochs * len(train_dataloader)
            lr_scheduler = get_scheduler(
                name="linear", optimizer=optimizer,
                num_warmup_steps=0, 
                num_training_steps=num_training_steps
            )
        elif args.exp == 'sgd':
            model.to(device)
            optimizer = SGD(model.parameters(), lr=args.sgdlr, weight_decay=1e-5, momentum=0.9)
        tic = 0

        if args.exp == 'fish':
            trail_info = "lr {:.6f}, auxlr {:.6f} sgdlr {:.5f} damping {:.4f} pretrain {:d}".format(lr, auxlr, sgdlr, damping, pretrain)
            print("difference:", args.diff, "norm:", args.norm)
        elif args.exp == 'adam':
            trail_info = 'lr {:.6f}, weight_decay {:.6f} adamW'.format(0.00005, args.weightdecay)

        print(trail_info)
        with open(result_file, 'a') as file_save:
            file_save.write(trail_info + '\n')
        if True:
            iteration = 0
            if args.exp == 'fish':
                if args.pretrain > 0: 
                    aux_losses = optimizer.pretrain_fish(
                                    args.pretrain,
                                    corpus_dataloader,
                                    nll,
                                    difference=args.diff,
                                    verbose=True
                            )
                    with open(result_file, 'a') as file_save:
                            file_save.write(','.join([str(a) for a in aux_losses])) 

            for epoch in range(args.num_epochs):

                for it, batch in enumerate(train_dataloader):

                    st = time.time()
                    batch = {k: v.to(device) for k, v in batch.items()}
                    optimizer.zero_grad()
                    loss = nll_movie(model, batch)
                    loss.backward()

                    optimizer.step()
                    if args.exp == 'fish':
                        free_opt.step()
                        free_scheduler.step()
                        free_opt.zero_grad()
                    else: 
                        lr_scheduler.step()
                    tic += (time.time() - st)


                    if iteration % args.save_every == 0:
                        eval_loss, eval_acc = 0,0
                        model.eval()
                        eval_dataloader = DataLoader(dataset["test"], shuffle=False, batch_size=32)
                        for _ in range(250):
                            test_batch = next(iter(eval_dataloader))
                            test_batch = {k: v.to(device) for k, v in batch.items()}

                            evaluate_result = evaluate(model, test_batch)
                            eval_loss += evaluate_result[0].detach().cpu().numpy()
                            eval_acc += evaluate_result[1]
                        eval_loss, eval_acc = eval_loss/250, eval_acc/250
                        eval_perp = perplexity(model).detach().cpu().numpy()

                        model.train()
                        train_loss = loss.detach().cpu().numpy()

                        if args.exp != 'fish':
                            print("epoch {:d}, it {:d}, train loss {:.4f}, \t eval loss {:.4f}, eval acc {:.4f}, \t perp {:.4f} lr {:.6f}".format(
                                epoch, iteration, train_loss, eval_loss, eval_acc, eval_perp, lr_scheduler.get_last_lr()[0])
                        )
                        else:
                            print("epoch {:d}, it {:d}, train loss {:.4f}, \t eval loss {:.4f}, eval acc {:.4f}, \t perp {:.4f}".format(
                                epoch, iteration, train_loss, eval_loss, eval_acc, eval_perp)
                        )

                        with open(result_file, 'a') as file_save:
                            file_save.write(','.join([str(elem) for elem in [
                                epoch, iteration, tic, train_loss, eval_loss, eval_acc, eval_perp
                            ]]))
                            file_save.write('\n')
                    elif iteration % args.print_every == 0:
                        train_loss = loss.detach().cpu().numpy()

                        print("epoch {:d}, it {:d}, train loss {:.4f}".format(
                            epoch, iteration, train_loss)
                        )
                    iteration += 1

        del model
        del optimizer
