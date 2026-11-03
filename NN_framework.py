import torch
import os
import torch.nn as nn
import numpy as np
from torch.nn import L1Loss
from MLP_Model import dag_structural_mlp_model, mlp_model, LSTM_model 
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import TensorDataset, DataLoader
from typing import Any, Dict, Optional, Tuple

from torch.amp import autocast, GradScaler


from process_model_calibration_manager import ProcessModelCalibrationManager
#from dag_masked_data import resolve_dag_model_config

from utilis_shorts import json_open


class NN_module(nn.Module):
    '''
    This class is used to control the models, training and so on
    '''
    def __init__(self, config) -> None:
        super(NN_module, self).__init__()
        if str(config.get('model::modeltype', '')).strip().lower() == 'dag_structural_mlp':
            self.config = resolve_dag_model_config(config)
        else:
            self.config = config

        #### cuda ####
        if self.config['training::cuda'] == True:
            if  torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        #### seed ####
        # Weight init must follow the *drawn* seed. `model::seed` is what Optuna
        # suggests per trial and what gets written back into the saved config;
        # `training::seed` is only the fixed fallback for runs that never draw one.
        # Reseeding with `training::seed` here would silently pin every trial to the
        # same initialisation and make the drawn seed a no-op.
        seed = self.config.get('model::seed', self.config['training::seed'])
        self.config['model::seed'] = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        #### batch ####
        self.batch = self.config['training::batchsize']
        
        #### model architectures ####
        if self.config['model::modeltype'] == 'MLP':
            self.model = mlp_model(self.config['model::layer_number'], 
                                   input_size=self.config['model::input_size'], 
                                   output_size=self.config['model::output_size'],
                                   hidden_size=self.config['model::hidden_size'], 
                                   activation=self.config['model::activation'],
                                   dropout_list=self.config['model::dropout_list'],
                                   bias_list=self.config['model::bias_list'])
            self.model = self.model.to(self.device)

        elif self.config['model::modeltype'] == 'LSTM':
            self.model = LSTM_model(lst_depth= config['model::lstm_depth'], 
                                    layer_number=self.config['model::layer_number'],
                                    input_size=self.config['model::input_size'],
                                    output_size=self.config['model::output_size'], 
                                    # canonical key is 'model::hidden_lstm' (what Optuna writes);
                                    # 'model:hidden_lstm' is the legacy spelling in older configs
                                    hidden_size_lstm= config.get('model::hidden_lstm',
                                                                 config.get('model:hidden_lstm')),
                                    hidden_size_lin = self.config['model::hidden_size'] , 
                                    activation=self.config['model::activation'], 
                                    dropout_list=self.config['model::dropout_list'], 
                                    bias_list=self.config['model::bias_list'])
            self.model = self.model.to(self.device)

        elif self.config['model::modeltype'] == 'dag_structural_mlp':
            self.model = dag_structural_mlp_model(
                layer_number=self.config['model::layer_number'],
                input_size=self.config['model::input_size'],
                output_size=self.config['model::output_size'],
                hidden_size=self.config['model::hidden_size'],
                activation=self.config['model::activation'],
                dropout_list=self.config['model::dropout_list'],
                bias_list=self.config['model::bias_list'],
                root_input_cols=self.config['data::dag_input_union_cols'],
                node_order=self.config['graph::structural_node_order'],
                node_to_parent_cols=self.config['graph::structural_parent_cols'],
                target_order=self.config.get('data::dag_target_cols'),
            )
            self.model = self.model.to(self.device)
        
        # elif self.config['model::modeltype'] == 'embedding':
        #     # Embedding model is initialized lazily once a dataloader is available.
        #     # This avoids hard-wiring PRELES keys and allows BASFOR-specific wiring.
        #     self.model = None
        #     self._embedding_ready = False
        #     self._embedding_process_model = str(
        #         self.config.get("process_model::name", "basfor")
        #     ).strip().lower()
        #     self._embedding_dataset = self.config.get("embedding::dataset")
        #     self.accum_steps = int(self.config.get("embedding::accum_steps", getattr(self, "accum_steps", 16)))
        #     self.use_amp = bool(self.config.get("embedding::use_amp", getattr(self, "use_amp", True)))
        #     self.scaler = GradScaler(enabled=(self.use_amp and self.device.type == "cuda"))
        
        # elif self.config['model::modeltype'] == 'Adapted_error' or self.config['model::modeltype'] == 'Symb_Adapted_error':
        #     self.lambda_ = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        #     self.optimizer_lambda =  Adam([self.lambda_], self.config['model:adapted::lambda_learning_rate']) 
        #     if self.config['model:error::modeltype'] == 'MLP':
        #         self.model_error = mlp_model(self.config['model:error::layer_number'], 
        #                            input_size=self.config['model:error::input_size'], 
        #                            output_size=self.config['model:error::output_size'],
        #                            hidden_size=self.config['model:error::hidden_size'], 
        #                            activation=self.config['model:error::activation'],
        #                            dropout_list=self.config['model:error::dropout_list'],
        #                            bias_list=self.config['model:error::bias_list'])
        #         self.model_error = self.model_error.to(self.device)
    
        #         if self.config['model::modeltype'] == 'Symb_Adapted_error':
        #             self.symb_model = []
        #     if self.config['model:real::modeltype'] == 'MLP':
        #         self.model_real = mlp_model(self.config['model::layer_number'], 
        #                             input_size=self.config['model::input_size'], 
        #                             output_size=self.config['model::output_size'],
        #                             hidden_size=self.config['model::hidden_size'], 
        #                             activation=self.config['model::activation'],
        #                             dropout_list=self.config['model::dropout_list'],
        #                             bias_list=self.config['model::bias_list'])
        #         self.model_real = self.model_real.to(self.device)

        #     elif self.config['model:real::modeltype'] == 'LSTM':
        #         self.model_real = LSTM_model(lst_depth= config['model::lstm_depth'], 
        #                             layer_number=self.config['model::layer_number'],
        #                             input_size=self.config['model::input_size'],
        #                             output_size=self.config['model::output_size'], 
        #                             hidden_size_lstm= config['model:hidden_lstm'],
        #                             hidden_size_lin = self.config['model::hidden_size'] , 
        #                             activation=self.config['model::activation'], 
        #                             dropout_list=self.config['model::dropout_list'], 
        #                             bias_list=self.config['model::bias_list'])
        #         self.model_real = self.model_real.to(self.device)
        #     else:
        #         raise NotImplementedError(f"Model type '{self.config['model::modeltype']}' not implemented.")
            
        else:
            raise NotImplementedError(f"Model type '{self.config['model::modeltype']}' not implemented.")
        
        #### Loss type ###
        if self.config['training::loss_type'] == 'MSE':
            self.loss= nn.MSELoss().to(self.device)
        elif self.config['training::loss_type'] == 'MAE':
            self.loss= nn.L1Loss().to(self.device)
        else:
            raise NotImplementedError(f"Loss type '{self.config['training::loss_type']}' not implemented.")

        loss_calculation = self.config['training::loss_physic_type']
        mtype = self.config['model::modeltype']

        #### Model Type training strategy ####
        # if mtype == 'embedding' and loss_calculation == 'embedding':
        #     self.train_epoch = self.train_embed_epoch
        #     self.test_epoch  = self.test_epoch_embed

        if loss_calculation in ('naive', 'bias', 'residual') and mtype != 'embedding':
            self.train_epoch = self.train_naive_epoch
            self.test_epoch  = self.test_epoch_naive

        elif loss_calculation == 'parallel':
            self.train_epoch = self.train_parallel_epoch
            self.test_epoch  = self.test_epoch_parallel

        elif loss_calculation == 'regularized':
            self.train_epoch = self.train_regularized_epoch
            self.test_epoch  = self.test_epoch_regularized
            self.val_epoch  = self.val_epoch_regularized

        elif loss_calculation == 'error':
            self.train_epoch = self.train_error_model
            self.test_epoch  = self.test_error_model

        else:
            raise NotImplementedError(
                f"Unsupported combination: loss={loss_calculation!r}, modeltype={mtype!r}"
            )
        
        #### Optimizer #### 
        if self.config['training::optimizer']=='SGD':
            self.optimizer = SGD(self.model.parameters(), 
                                lr=self.config['training::learning_rate'],
                                momentum=self.config['training::momentum'],
                                weight_decay=self.config['training::weight_decay'], 
                                dampening=self.config['training::dampening'])
        
        elif self.config['training::optimizer']=='Adam' and self.config['model::modeltype']:
            self.optimizer = Adam(self.model.parameters(), 
                                lr=self.config['training::learning_rate'],
                                betas= (self.config['training::beta_1'],
                                        self.config['training::beta_2']),
                                weight_decay=self.config['training::weight_decay'])
        elif self.config['training::optimizer']=='AdamW' and self.config['model::modeltype']:
            self.optimizer = AdamW(self.model.parameters(), 
                                lr=self.config['training::learning_rate'],
                                betas= (self.config['training::beta_1'],
                                        self.config['training::beta_2']),
                                weight_decay=self.config['training::weight_decay'])
        else:
            raise NotImplementedError(f"Optimizer '{self.config['training::optimizer']}' not implemented.")
    
        #### metrics ####
        if self.config['training::acc_metric'] == 'MAE':
            self.metric = L1Loss().to(self.device) 
        elif self.config['training::acc_metric'] ==  'MSE':
            self.metric= nn.MSELoss().to(self.device)


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def forward(self, x):
        # compute model prediction
        y = self.model(x)
        return y
        
    def save_model(self,epoch):
        dict={}
        if self.config['model::save_path'] is not None:
            fname = os.path.join(f"model_epoch_{epoch}.ptf")
            dict["state_dict"] = self.model.state_dict()
            torch.save(dict, fname)
        return None

    def _make_optimizer(self, param_groups):
        opt_name = str(self.config['training::optimizer']).strip()
        if opt_name == 'SGD':
            return SGD(
                param_groups,
                momentum=self.config['training::momentum'],
                weight_decay=self.config['training::weight_decay'],
                dampening=self.config['training::dampening'],
            )
        if opt_name == 'Adam':
            return Adam(
                param_groups,
                betas=(self.config['training::beta_1'], self.config['training::beta_2']),
                weight_decay=self.config['training::weight_decay'],
            )
        if opt_name == 'AdamW':
            return AdamW(
                param_groups,
                betas=(self.config['training::beta_1'], self.config['training::beta_2']),
                weight_decay=self.config['training::weight_decay'],
            )
        raise NotImplementedError(f"Optimizer '{opt_name}' not implemented.")

    ### Naive & Bias ### 
    def train_naive_epoch(self, dataloader, epoch_index):
        '''plain training step'''
            # Set the model to training mode - important for batch normalization and dropout layers
            # Unnecessary in this situation but added for best practices
        #size = len(dataloader.dataset)
        self.model.train()
        avag_loss = 0
        avag_metric= 0 

        dataloader= dataloader
        # Weight each batch by its sample count. `self.loss` reduces with mean, so
        # summing batch means and dividing by the batch count gives the mean of the
        # batch means — which differs from the mean over samples whenever the final
        # batch is a short remainder (drop_last is False, so it usually is).
        n_samples = 0
        for X, y in dataloader:
            # X = X.to(self.device)
            # y = y.to(self.device)

            self.optimizer.zero_grad()
            # Compute prediction,loss and its gradient
            pred = self.model(X)
            loss_ = self.loss(pred, y)
            loss_.backward()
            # Adjust learning weights
            self.optimizer.step()
            #evaluating state
            last_metric= self.metric(pred,y)
            # epochs
            batch_n = y.size(0)
            avag_loss+=loss_ * batch_n
            avag_metric+=last_metric * batch_n
            n_samples += batch_n

        # print(f"Train Error: {(avag_metric/len(dataloader)):>8f}, Avg loss: {(avag_loss/len(dataloader)):>8f} \n")
        return avag_loss/max(n_samples, 1), avag_metric/max(n_samples, 1) #average loss of each epoch

    def test_epoch_naive(self, dataloader):
        #Set the model to evaluation mode - important for batch normalization and dropout layers
        self.model.eval()
        test_loss, metric = 0, 0
        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # Sample-weighted: see the note in train_naive_epoch.
        n_samples = 0
        with torch.no_grad():
            for X, y in dataloader:
                # X = X.to(self.device)
                # y = y.to(self.device)
                pred = self.model(X)
                batch_n = y.size(0)
                test_loss += self.loss(pred, y).item() * batch_n
                metric += self.metric(pred,y) * batch_n
                n_samples += batch_n
        test_loss /= max(n_samples, 1)
        metric /= max(n_samples, 1)
        # print(f"Test Error:{(metric):>8f}, Avg loss: {test_loss:>8f} \n")
        return test_loss, metric

    ### Parallel ###
    def train_parallel_epoch(self, dataloader_physical, epoch_index):
        '''train_parallel step'''
        # size = len(dataloader_physical.dataset)
            # Set the model to training mode - important for batch normalization and dropout layers
        self.model.train(True)
        avag_loss = 0
        avag_metric= 0 
        last_metric=0
        last_batch_metric=0
        last_batch_loss=0
        
        # Sample-weighted: see the note in train_naive_epoch.
        n_samples = 0
        for i, (X,y,y_p) in enumerate(dataloader_physical):
            self.optimizer.zero_grad()

            # Compute prediction and loss
            # print(f"[DEBUG] {self.model(X).shape}, {y_p.shape}")
            pred = self.model(X) + y_p # todo, adapt to batchsize/calling
            loss = self.loss(pred, y)
            # Backpropagation
            loss.backward()
            #Adjust learning weights
            self.optimizer.step()

            #evaluating performance
            last_metric+= self.metric(pred,y)
            batch_n = y.size(0)
            avag_loss+=loss * batch_n
            #evaluating state
            last_metric= self.metric(pred,y)
            avag_metric+=last_metric * batch_n
            #batches
            last_batch_metric+= last_metric
            last_batch_loss+= loss
            n_samples += batch_n
            if i % 10 == 9:
                current = (self.batch + 1) *i
                #current state within training epoch
                #print(f"epoch:{epoch_index:>5d} batch loss: {last_batch_loss.item():>7f} batch metric:{last_batch_metric.item():>7f}    [{current:>5d}/{size:>5d}]")
                last_batch_metric=0
                last_batch_loss=0

        return avag_loss/max(n_samples, 1), avag_metric/max(n_samples, 1)

    def test_epoch_parallel(self,dataloader):
        #Set the model to evaluation mode - important for batch normalization and dropout layers
        self.model.eval()
        test_loss, metric = 0, 0
        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # Sample-weighted: see the note in train_naive_epoch.
        n_samples = 0
        with torch.no_grad():
            for X, y, y_p in dataloader:
                pred = self.model(X)+ y_p
                batch_n = y.size(0)
                test_loss += self.loss(pred, y).item() * batch_n
                metric += self.metric(pred,y) * batch_n
                n_samples += batch_n
        test_loss /= max(n_samples, 1)
        metric /= max(n_samples, 1)
        #print(f"Test Error: \n Accuracy: {(metric):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss, metric

    ### Regularised ###
    def train_regularized_epoch(self,dataloader_reg, epoch_index):
        '''train_regularized step'''
        #size = len(dataloader_reg.dataset)
            # Set the model to training mode - important for batch normalization and dropout layers
            # Unnecessary in this situation but added for best practices
        self.model.train()
        avag_loss = 0
        avag_metric= 0 
        metric = 0
        last_batch_metric=0
        last_batch_loss=0
        lamb = self.config['training::physical_loss_regul_term']

        # Sample-weighted: see the note in train_naive_epoch.
        n_samples = 0
        for i, (X, y,y_p) in enumerate(dataloader_reg):
            # Compute prediction and loss
            pred = self.model(X) # todo, adapt to batchsize/calling
            loss = self.loss(pred, y)+ lamb*self.loss(y_p,pred)
            metric+= self.metric(pred,y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            #evaluating metrics
            last_metric= self.metric(pred,y)
            #epochs
            batch_n = y.size(0)
            avag_loss+=loss * batch_n
            avag_metric+=last_metric * batch_n
            #batches
            last_batch_metric+= last_metric
            last_batch_loss+= loss
            n_samples += batch_n
            if i % 10 == 9:
                current = (self.batch + 1) *i
                #current state within training epoch
                #print(f"epoch:{epoch_index:>5d} batch loss: {last_batch_loss.item():>7f} batch metric:{last_batch_metric.item():>7f}    [{current:>5d}/{size:>5d}]")
                last_batch_metric=0
                last_batch_loss=0

        return avag_loss/max(n_samples, 1), avag_metric/max(n_samples, 1)

    def test_epoch_regularized(self,dataloader):
        #Set the model to evaluation mode - important for batch normalization and dropout layers
        self.model.eval()
        test_loss, metric = 0, 0
        lamb = self.config['training::physical_loss_regul_term']
        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # Sample-weighted: see the note in train_naive_epoch.
        n_samples = 0
        with torch.no_grad():
            for X, y, _ in dataloader:
                pred = self.model(X)
                batch_n = y.size(0)
                test_loss += self.loss(pred, y).item() * batch_n
                metric += self.metric(pred,y).item() * batch_n
                n_samples += batch_n
        test_loss /= max(n_samples, 1)
        metric /= max(n_samples, 1)
        #print(f"Test Error: \n Accuracy: {(metric):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss, metric

    def val_epoch_regularized(self,dataloader):
        #Set the model to evaluation mode - important for batch normalization and dropout layers
        self.model.eval()
        val_loss, metric = 0, 0
        lamb = self.config['training::physical_loss_regul_term']
        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # Sample-weighted: see the note in train_naive_epoch.
        n_samples = 0
        with torch.no_grad():
            for X, y, _ in dataloader:
                pred = self.model(X)
                batch_n = y.size(0)
                val_loss += self.loss(pred, y).item() * batch_n
                metric += self.metric(pred,y).item() * batch_n
                n_samples += batch_n
        val_loss /= max(n_samples, 1)
        metric /= max(n_samples, 1)
        #print(f"Test Error: \n Accuracy: {(metric):>0.1f}%, Avg loss: {val_loss:>8f} \n")
        return val_loss, metric
    
    ### Error ###
    def train_error_model(self,error_dataloader, epoch):

        loss_all, metric_all = 0,0
        self.model.train()

        n_batches = 0
        for X, _, y_p in error_dataloader:
            if self.config['model::modeltype'] == 'LSTM':
                x= torch.cat((X[0], X[1]),2)
            else:
                x= torch.cat((X[0], X[1]),1)
            pred= self.model(x) #Do I also use real input data or only Process data?
            loss = self.loss(pred, y_p[1])
            loss.backward()
            self.optimizer.zero_grad()
            self.optimizer.step()

            loss_all += loss.item()
            metric_all += self.metric(pred, y_p[1])
            n_batches += 1

        return metric_all/max(n_batches, 1), loss_all/max(n_batches, 1)

    def test_error_model(self, error_dataloader):
        self.model.eval()
        test_loss, metric = 0,0
      
        n_batches = 0
        for X, _, y_p in error_dataloader:
            if self.config['model::modeltype'] == 'LSTM':
                x= torch.cat((X[0], X[1]),2)
            else:
                x= torch.cat((X[0], X[1]),1)
            pred= self.model(x) #Do I also use real input data or only Process data?
            loss = self.loss(pred, y_p[1])

            test_loss += loss.item()
            metric += self.metric(pred, y_p[1])
            n_batches += 1

        return metric/max(n_batches, 1), test_loss/max(n_batches, 1)
    
    ### Embedding ###
    def train_embedding_epoch(self,dataloader, epoch):
        loss_all, metric_all = 0,0
        self.model.train()
        
        device = self.device
        scaler = self.scaler
        step_in_accum = 0
        num_batches = 0
        
        self.optimizer.zero_grad(set_to_none=True)
        for x, y in dataloader:
            x, y = self._prepare_embedding_batch(x, y)
            
            if scaler.is_enabled():
                with autocast(device_type="cuda", enabled=True):
                    y_pred = self.model(x, y)
                    loss = self.loss(y_pred, y) / self.accum_steps
            else:    
                y_pred = self.model(x, y)
                loss = self.loss(y_pred, y) / self.accum_steps

            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                step_in_accum = 0
                continue
        
            # backward
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            
            # logs
            loss_all += float(loss.detach()) * self.accum_steps
            metric_all += float(self.metric(y_pred, y).detach())
            num_batches += 1
            step_in_accum += 1

            # step every accum_steps
            if step_in_accum == self.accum_steps:
                # if self.clip_grad_norm is not None:
                #     if scaler.is_enabled():
                #         scaler.unscale_(self.optimizer)
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                if scaler.is_enabled():
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
                if hasattr(self.model, "process_layer") and hasattr(self.model.process_layer, "clamp_if_needed"):
                    self.model.process_layer.clamp_if_needed()

                self.optimizer.zero_grad(set_to_none=True)
                step_in_accum = 0

        # handle remainder
        if step_in_accum > 0:
            # if self.clip_grad_norm is not None:
            #     if scaler.is_enabled():
            #         scaler.unscale_(self.optimizer)
            #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            if scaler.is_enabled():
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()
            if hasattr(self.model, "process_layer") and hasattr(self.model.process_layer, "clamp_if_needed"):
                self.model.process_layer.clamp_if_needed()
            self.optimizer.zero_grad(set_to_none=True)
            

        avg_metric = metric_all / max(num_batches, 1)
        avg_loss   = loss_all   / max(num_batches, 1)
        return avg_metric, avg_loss

    def test_epoch_embed(self,dataloader):
        self.model.eval()
        test_loss, metric = 0, 0

        n_batches = 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = self._prepare_embedding_batch(x, y)

                pred = self.model(x,y)
                test_loss += self.loss(pred, y).item()
                metric += self.metric(pred,y)
                n_batches += 1

        test_loss /= max(n_batches, 1)
        metric /= max(n_batches, 1)
        #print(f"Test Error: \n Accuracy: {(metric):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss, metric


    


def _to_scalar(v):
        return float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v)

if __name__=='__main__':
    from Config_Files.config_spatial import build_config
    from Config_Files.config_loader import merge_process_model_config
    IDE_SETTINGS = {
        # Choose what to run from IDE: "dataset" (full pipeline) or "smoke" (quick synthetic sanity check)
        "mode": "dataset",
        "config_name": "spatial",  # "spatial" | "embedding"
        "calibrate": True,
        "embedding": {
            "target_names": ["GPP", "ET", "NEE", "SWC"],
            "time_batching": True,
            "shuffle_train": False,
            "param_mode": "factor",  # "factor" | "bounded"
            "factor_min": 0.2,
            "factor_max": 5.0,
            "pm_grad_sign": 1.0,
            "use_output_head": False,
            "accum_steps": 1,
            "use_amp": False,
        },
        "smoke": {
            "n_samples": 256,
            "epochs": 8,
            "batchsize": 32,
            "seed": 42,
        },
    }


    base_config = build_config()
    merged_config = merge_process_model_config(
        base_config=base_config,
        process_model="preles",
    )

    calibrate = bool(IDE_SETTINGS.get("calibrate", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device} | calibrate={calibrate}")
    module = NN_module(merged_config).to(device)

    train_stations = dict(merged_config["data::train::station_id"])
    test_stations = dict(merged_config["data::test::station_id"])
    val_stations = dict(merged_config["data::val::station_id"]) if "data::val::station_id" in merged_config else None

    manager = ProcessModelCalibrationManager(config=merged_config, process_model=merged_config.get('process_model::name'))
    
    train_loader, test_loader, val_loader = manager.load_fold_dataloaders(
        train_stations=train_stations,
        test_stations=test_stations,
        val_stations=val_stations,
        calibrate=calibrate,
        shuffle_train=False,
        sparse=None,
    )

    for i in range(merged_config['training::epochs']):
        train_loss, train_metric = module.train_epoch(train_loader, i)
        test_loss, test_metric = module.test_epoch(test_loader)
        # print(
        #     f"[EPOCH {i+1:03d}] "
        #     f"train_loss={_to_scalar(train_loss):.6f} train_metric={_to_scalar(train_metric):.6f} "
        #     f"test_loss={_to_scalar(test_loss):.6f} test_metric={_to_scalar(test_metric):.6f}"
        # )
    
