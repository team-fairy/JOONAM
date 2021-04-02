from torch.optim import Adam
from torch.optim import SGD
import tqdm as tqdm
from tqdm import tqdm_notebook as tq
import time

# model load
class Trainer:
    def __init__(model, model_name,transform, dataset, optimizer,learning_rate, weight_decay, batch_size, train_ratio, loss_function ):
        self.model = model()
        self.transform = transform
        self.dataset = dataset(transform)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.criterion = loss_function


        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        len_train_set = int( self.train_ratio * len(dataset) )
        len_val_set = len(dataset) - len_train_set

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset,
                                                                [len_train_set, len_val_set])

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers = 4)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers = 4)
        self.model.to(self.device)

        self.log = f"""{model_name}| batch : {batch_size} | 
        \n| loss function : {loss_function}
        \n| optimizer : {optimizer} | weight_decay : {weight_decay} | learning_rate : {learning_rate} 
        \n| normalize : True | size = {input_size}"""


    def train_one_epoch(epoch):
        self.model.train()
        
        train_loss = 0
        correct = 0
        total = 0
        load_time = 0
        inference_time = 0
        
        self.log += f"\n\n====train model epoch : {epoch+1}====\n"  
        print(f"\n\n====train model epoch : {epoch+1}====")
        start_time = time.time()
        
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            s = time.time()
            self.optimizer.zero_grad() # 각각의 모델 파라미터 안의 gradient가 update 된다. 
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            inference_time += time.time() - s
            load_time = time.time() - start_time - inference_time
            
            if batch_idx % 10 == 0:
                self.log +=  "batch : %.3d | Loss: %.3f | Acc : %.3f%% | inference_time : %.3f / load_time : %.3f \n" \
                    %(batch_idx,train_loss/(batch_idx+1),100.*correct/total, inference_time/(batch_idx+1), load_time/(batch_idx+1e-04))
                print( "batch : %.3d | Loss: %.3f | Acc : %.3f%% | inference_time : %.3f / load_time : %.3f \r" \
                    %(batch_idx,train_loss/(batch_idx+1),100.*correct/total, inference_time/(batch_idx+1), load_time/(batch_idx+1e-04)), end = "" )

                
    def test_one_opoch(epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        print()
        print(f"====Evaluation model epoch : {epoch+1}====")
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                print( f"progress:{batch_idx*BATCH_SIZE} | {len_val_set} \r", end = "" )
        self.log += f'Accuracy of the network on the {total} test image : %d %%' % ( 100*correct/total )    
        print(f'Accuracy of the network on the {total} test image : %d %%' % ( 100*correct/total ))
        

    def save_model(epoch, save_path, name = 'student'):
        print(f'saved model {save_path}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = f"{name}_{epoch}.pth"
        file_path = os.path.join(save_path, file_name)
        torch.save(self.model.state_dict(), file_path)
        with open(f'{save_path}/log.txt' , 'w') as f:
            f.write(self.log)
        return None