import torch

class Aggregator(torch.nn.Module):
    def __init__(self, num_clients):
        super(Aggregator, self).__init__()
        self.num_clients = num_clients
        self.weight = torch.nn.Parameter(torch.tensor([0.04569443, 0.29310702, 0.12821656, 0.0888967,  0.01651702, 0.01651424, 0.00582673, 0.19584123, 0.08949454, 0.11989152]), requires_grad=True)
        
    def forward(self, current_client_vars_dict):
        client_vars_sum = None
        for c_id in range(self.num_clients):
            if client_vars_sum is None:
                client_vars_sum = dict((key, (value * self.weight[c_id])) for key, value in current_client_vars_dict[c_id].items())
            else:
                for key, value in current_client_vars_dict[c_id].items():
                    client_vars_sum[key] = client_vars_sum[key] + (value * self.weight[c_id])
                    
        return client_vars_sum
    
    def get_global_model(self, current_client_vars_dict):
        client_vars_sum = None
        for c_id in range(self.num_clients):
            if client_vars_sum is None:
                client_vars_sum = dict((key, (value * self.weight[c_id]).detach()) for key, value in current_client_vars_dict[c_id].items())
            else:
                for key, value in current_client_vars_dict[c_id].items():
                    client_vars_sum[key] = client_vars_sum[key] + (value * self.weight[c_id]).detach()
                    
        return client_vars_sum