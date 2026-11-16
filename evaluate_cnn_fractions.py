"""Final preplanned test comparison of CNNs trained with different data fractions."""
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from project_config import RESULTS_DIR, CLASSES
from data_utils import load_radioml, build_arrays
from train_cnn import SmallAMC


def main():
    torch.set_num_threads(4)
    data=load_radioml(); X,y,snr=build_arrays(data)
    splits=np.load(RESULTS_DIR/'fixed_splits.npz')
    results={"note":"Preplanned training-size sensitivity analysis. Test data were not used for selecting checkpoints.","models":{}}
    for fraction, path in [(0.25,RESULTS_DIR/'cnn_fraction_25.pt'),(0.50,RESULTS_DIR/'cnn_fraction_50.pt'),(1.00,RESULTS_DIR/'cnn_best.pt')]:
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
        model=SmallAMC(len(CLASSES)); model.load_state_dict(ckpt['model_state_dict']); model.eval()
        scale=float(ckpt['rms_scale'])
        key=f"{int(fraction*100)}%"
        results['models'][key]={"training_observations":int(ckpt.get('train_count',len(splits['train']) if fraction==1.0 else round(len(splits['train'])*fraction))),"best_epoch":int(ckpt['best_epoch']),"validation_macro_f1":float(ckpt['validation_macro_f1'])}
        for part in ('familiar_test','unseen_snr_test'):
            idx=splits[part]; xt=torch.from_numpy((X[idx].copy()/scale).astype(np.float32));
            pred=[]
            with torch.no_grad():
                for start in range(0,len(xt),2048): pred.extend(model(xt[start:start+2048]).argmax(1).numpy().tolist())
            yhat=np.asarray([CLASSES[i] for i in pred]); yt=y[idx]
            results['models'][key][part]={"observations":int(len(idx)),"accuracy":float(accuracy_score(yt,yhat)),"macro_f1":float(f1_score(yt,yhat,labels=list(CLASSES),average='macro'))}
        print(key,results['models'][key])
    (RESULTS_DIR/'cnn_training_size_test.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__': main()
