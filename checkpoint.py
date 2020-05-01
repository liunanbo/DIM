import os
import io
import torch
from datetime import datetime
import socket
from pathlib import Path
from Model import Encoder,DeepInfoMaxLoss,get_models
from viola.db import models
from viola.config.config import CONNECTION_MANAGER_PROXY, config
from copy import copy
from viola.db.models import Run, Model
from viola.utils.io import s3_to_tempfile


class Checkpointer():
    def __init__(self, run ):
        # set output dir will this checkpoint will save itself
        self.config =run.config
        self.version = self.config['train']['version']
        self.input_shape= self.config['model']['net_kwargs']['input_shape']
        self.n_epoch = 0
        self.run = run
        bucket_name = self.config['trainer_kwargs']['bucket_name']
        _,self.bucket = CONNECTION_MANAGER_PROXY.get_s3_rsrc_and_bucket(bucket_name=bucket_name)


    # get networks state dictionary
    def _get_state(self):
        return {
            'encoder': self.encoder.state_dict(),
            'mi_estimator':self.mi_estimator.state_dict(),
            'n_epoch':self.n_epoch
        }


    def _save_cpt(self,notes=None,name=None):
        CONNECTION_MANAGER_PROXY.connect_to_mongo(config["mongodb"])
        mod = models.Model(
            type=self.config['type'], notes=notes, name=name, run=self.run, ckpt=self.n_epoch
        ).save()
        # DIM/<model Version>/<model checkpoint>
        path = "DIM/{}/{}-{}.pth".format(self.version, mod.pk, self.n_epoch)
        #update run checkpoints
        self.run.opt_ckpts[str(self.n_epoch)] = path
        #update model configure
        self.config['net_path']=path
        mod.config = self.config
        mod.save()
        self.run.save()
        
        # Save model
        # write updated checkpoint to the desired path
        f = io.BytesIO()
        torch.save(self._get_state(),f)
        f.seek(0)
        self.bucket.upload_fileobj(f, path)


        print("***** SAVEPOINT *****\n"
        "Version: {}\n"
        "Model is saved at epoch {}.\n"
        "*************************"
        .format(self.version,self.n_epoch))
        return


    def update(self, save_interval):

        self.n_epoch += save_interval
        self._save_cpt()


    
    def restore_model_from_checkpoint(self):

        self.encoder,self.mi_estimator = get_models(**self.config['model']['net_kwargs'])
        ckpts = list(Model.objects(run=self.run).scalar("ckpt"))
        if len(ckpts)>0:
            #Load model, collect the latest checkpoint
            ckpt = max(ckpts)
            path = self.run.opt_ckpts[str(ckpt)]
            ckpt_temp = s3_to_tempfile(bucket_name=self.bucket.name,path=path)
            ckpt_temp.seek(0)
            model_ckpt = torch.load(ckpt_temp,map_location='cpu')        


            encoder_params = model_ckpt['encoder']
            mi_estimator_params = model_ckpt['mi_estimator']
            self.n_epoch = model_ckpt['n_epoch']

            self.encoder.load_state_dict(encoder_params)
            self.mi_estimator.load_state_dict(mi_estimator_params)


            print("***** CHECKPOINT *****\n"
                    "Version: {}\n"
                    "Model restored from checkpoint.\n"
                    "Self-supervised training from epoch {}\n"
                    "*************************"
                    .format(self.version,self.n_epoch))
        else:
            print('There are no checkpoints. Construct new models.')

        return self.encoder,self.mi_estimator







