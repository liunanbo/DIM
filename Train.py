import argparse
import os
from tqdm import tqdm
from torchvision import transforms, utils
from Model import Encoder,DeepInfoMaxLoss,get_models
import torch 
from checkpoint import Checkpointer
import numpy as np
from mongoengine import connect
from viola.config import config
import json
from Loader import Image_Loader
from viola.db.models import Run, Model
from datetime import datetime
import socket
from copy import deepcopy



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_id', type=int, 
                    help='Mongodb run id')

    parser.add_argument('--img_dir',type=str,dest='img_path',
                    help='directory where images are stored')


    parser.add_argument('--gpu_ids', type=str, default='13',
                    help='Concatenated multiple GPU Devices Index (If use Multiple GPUs #1,#2,#3, then index should be 123)')



    args = parser.parse_args()


    gpu_ind = (",").join([i for i in args.gpu_ids])
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_ind



    #Collect run from run id
    connect(**config["mongodb"])
    run_id = args.run_id
    run = Run.objects.get(pk=run_id)
    run.status = 'RUNNING'
    run.hostname = socket.gethostname()
    run.pid = os.getpid()
    run.starttime = datetime.now()
    run.save()


    # Load model config file 
    model_conf = deepcopy(dict(run.config))

    # batch size must be an even number
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_path = args.img_path

    batch_size = model_conf['train_data_loader_kwargs']["batch_size"]
    num_epochs = model_conf['train']['n_global_steps']
    input_shape = model_conf['model']['net_kwargs']['input_shape']
    save_interval = model_conf['train']['global_step_save_ckpt_freq']
    num_workers = model_conf['train_data_loader_kwargs']['num_workers']
    lr = model_conf['trainer_kwargs']['opt_kwargs']['lr']

    #update run id in config file
    model_conf['run_id']=run_id
    json.dump(model_conf,open('config.json','w'))

    #Construct DataLoader and checkpointer
    train_loader  = Image_Loader(img_path,batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=num_workers,
                                                input_shape=input_shape,
                                                stage='train')
    checkpointer = Checkpointer(run= run)

    # Load checkpoint if given, otherwise construct a new model
    encoder, mi_estimator = checkpointer.restore_model_from_checkpoint()

    # Compute on multiple GPUs, if there are more than one given
    if torch.cuda.device_count() > 1:
        print("Let's use %d GPUs" %torch.cuda.device_count())
        encoder = torch.nn.DataParallel(encoder).module
        mi_estimator = torch.nn.DataParallel(mi_estimator).module
    encoder.to(device)
    mi_estimator.to(device)

    enc_optim = torch.optim.Adam(encoder.parameters(), lr=lr)
    mi_optim = torch.optim.Adam(mi_estimator.parameters(), lr=lr)
    try:        
        encoder.train()
        mi_estimator.train()
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(1, num_epochs+1):
            Batch = tqdm(train_loader, total=len(train_loader))
            for i, data in enumerate(Batch, 1):
                data = data.to(device)

                #E is encoding, M is local feature map
                E, M = encoder(data)
                # shuffle batch to pair each element with another
                M_fake = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
                encoder_loss, mi_loss = mi_estimator(E, M, M_fake)

                Batch.set_description(f"[{epoch:>3d}/{num_epochs:<3d}]Loss/Train: {mi_loss.item():1.5e}")
                
                # Optimize encoder
                enc_optim.zero_grad()
                encoder_loss.backward(retain_graph=True)
                enc_optim.step()
                # Optimize mutual information estimator
                mi_optim.zero_grad()
                mi_loss.backward()
                mi_optim.step()

            # checkpoint and save models
            if epoch % save_interval == 0:
                checkpointer.update(save_interval)

    except KeyboardInterrupt:
        run.status = "STOPPED BY USER"
        run.save()
    
    finally:
        if run.status == 'RUNNING':
            run.status = "STOPPED"
        run.endtime = datetime.now()
        run.save()
























            
