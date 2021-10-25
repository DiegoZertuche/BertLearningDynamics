import base_model
import task_head
import torch
import torch.nn as nn


class MainModel(nn.Module):

    def __init__(self, model_params, task_params, is_cuda):
        super(MainModel, self).__init__()
        self.model_params = model_params
        self.task_params = task_params
        self.encoder = base_model.BaseModel(model_params)
        self.task_params['d_hidden'] = self.encoder.get_output_dim()
        self.head = task_head.EdgeClassifierModule(task_params)
        self.is_cuda = is_cuda
        if self.is_cuda:
            self.encoder.to('cuda')
            self.head.to('cuda')

    def forward(self, batch, predict):
        if self.is_cuda:
            hidden_states = self.encoder(batch[0].to('cuda'))
            batch_ = {'batch1': hidden_states, 'span1s': batch[1].to('cuda'), 'span2s': batch[2].to('cuda'),
                      'labels': batch[3].to('cuda')}
            outs = self.head(batch_, batch[0]['attention_mask'].to('cuda'), predict)
        else:
            hidden_states = self.encoder(batch[0])
            batch_ = {'batch1': hidden_states, 'span1s': batch[1], 'span2s': batch[2],
                      'labels': batch[3]}
            outs = self.head(batch_, batch[0]['attention_mask'], predict)
        return outs
