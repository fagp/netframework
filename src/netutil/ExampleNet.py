import torch
from .NetFramework import NetFramework
import torch.nn.functional as F

class ExampleNet(NetFramework):
    def __init__(self,args):
        NetFramework.__init__(self,args)

    def valid_visualization(self,current_epoch,index=0,save=False):   
        self.net.eval()   

        classes=['Izq','Derch']
        sample=self.testdataset[ index ]
        sample['image'].unsqueeze_(0)
        sample['label'].unsqueeze_(0)
        
        sample=self.warp_var_mod.warp_Variable(sample,self.use_cuda,grad=False)
        images=sample['image']

        outputs = self.net(images)       

        classific= torch.argmax(F.softmax(outputs[0],1))

        img=images[0].cpu().numpy()
        if self.visdom==True:
            self.visimshow.show('Image',img)
            self.vistext.show('Class','GT: '+classes[sample['label'][0,0].item()]+'\n'+'CL: '+classes[classific.item()]+'\n' )

        return 1