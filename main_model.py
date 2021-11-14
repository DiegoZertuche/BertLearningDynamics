import base_model
import task_head
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm


class MainModel(nn.Module):

    def __init__(self, model_params, task_params, device):
        super(MainModel, self).__init__()
        self.model_params = model_params
        self.task_params = task_params
        self.encoder = base_model.BaseModel(model_params)
        self.device = device
        self.task_params['d_hidden'] = self.encoder.get_output_dim()
        self.head = task_head.EdgeClassifierModule(self.task_params, self.device)
        if self.device == 'cuda':
            self.encoder.to('cuda')
            self.encoder.to('cuda')
        elif self.device == 'tpu':
            self.encoder.to(xm.xla_device())
            self.encoder.to(xm.xla_device())

    def forward(self, batch, predict):
        if self.device == 'cuda':
            hidden_states = self.encoder(batch[0].to('cuda'))
            batch_ = {'batch1': hidden_states, 'span1s': batch[1].to('cuda'), 'span2s': batch[2].to('cuda'),
                      'labels': batch[3].to('cuda')}
            outs = self.head(batch_, batch[0]['attention_mask'].to('cuda'), predict, self.device)

        elif self.device == 'tpu':
            hidden_states = self.encoder(batch[0].to(xm.xla_device()))
            batch_ = {'batch1': hidden_states,
                      'span1s': batch[1].to(xm.xla_device()),
                      'span2s': batch[2].to(xm.xla_device()),
                      'labels': batch[3].to(xm.xla_device())}
            outs = self.head(batch_, batch[0]['attention_mask'].to(xm.xla_device()), predict, self.device)

        else:
            hidden_states = self.encoder(batch[0])
            batch_ = {'batch1': hidden_states, 'span1s': batch[1], 'span2s': batch[2],
                      'labels': batch[3]}
            outs = self.head(batch_, batch[0]['attention_mask'], predict, self.is_cuda)
        return outs
