import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from pprint import pprint
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

def plot_compare(model, metrics, eval_results, epochs,
                 outfile=None):
    for m in metrics:
        train_score = eval_results['train'][m]
        val_score = eval_results['val'][m]
        rang = range(0, len(train_score))
        plt.rcParams["figure.figsize"] = [6,6]
        plt.plot(rang, val_score, "c", label="Val")
        plt.plot(rang, train_score, "orange", label="Train")
        plt.xlabel('Iterations')
        plt.ylabel(m)
        lgd = plt.legend()
        plt.tight_layout()
        if outfile is not None:
            fpath = os.path.splitext(outfile)[0]
            fpath = f'{fpath}-{m}.png'
            plt.savefig(fpath)
            print(f"Saved figure to {fpath}")
        else:
            plt.show()
        plt.clf()

def plot_feature_importances(model, outfile=None):
    features = [f"{i}" for i in range(X_train.shape[1])]
    f, ax = plt.subplots(figsize=(10,5))
    importances = xgb1.feature_importances_
    plot = sns.barplot(x=features, y=importances)
    ax.set_title('Feature Importance')
    plot.set_xticklabels(plot.get_xticklabels(),rotation='vertical')
    plt.tight_layout()
    if outfile is not None:
        fpath = os.path.splitext(outfile)[0]
        fpath = f'{fpath}-fi.png'
        plt.savefig(fpath)
        print(f"Saved figure to {fpath}")
    else:
        plt.show()
    plt.clf()

def fitXgb(model, X_train, y_train, X_val, y_val, X_test, y_test,
           epochs=300, outfile=None):
    print('Fitting model...')
    model.fit(X_train, y_train)
    print('Fitting done!')
    train = xgb.DMatrix(X_train, label=y_train)
    val = xgb.DMatrix(X_val, label=y_val)
    params = model.get_xgb_params()
    metrics = ['mlogloss', 'merror']
    params['eval_metric'] = metrics
    store = {}
    evallist = [(val, 'val'), (train,'train')]
    xgb_model = xgb.train(params, train, epochs, evallist,
                          early_stopping_rounds=10,
                          evals_result=store, verbose_eval=100)
    print('-- Model Report --')
    print('XGBoost Accuracy: ' + str(accuracy_score(model.predict(X_test), y_test)))
    print('XGBoost F1-Score (Micro): ' + str(f1_score(model.predict(X_test), y_test, average='micro')))
    if outfile is not None:
        pickl = {'model': model}
        pickle.dump(pickl, open(outfile, 'wb'))
        print(f"Wrote model to {outfile}")
    plot_compare(model, metrics, store, epochs, outfile=outfile)
    plot_feature_importances(model, outfile=outfile)
    return xgb_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # video options
    parser.add_argument('-d', '--dataset', default='', help='dataset')
    parser.add_argument('-e', '--nestimators', type=int, default=250)
    parser.add_argument('-m', '--max-depth', type=int, default=5)
    parser.add_argument('-c', '--cuda_device', type=int, default=0)
    parser.add_argument('-n', '--nepochs', type=int, default=300)
    parser.add_argument('-f', '--filter_na', action='store_true', help='whether to filter out class 18 (N/A)')
    parser.add_argument('-t', '--nthreads', type=int, default=2)
    args = parser.parse_args()

    print("parameters:")
    pprint(vars(args), indent=1)

    ds = args.dataset
    dsdir, dsfname = os.path.split(ds)
    dsname, _ = os.path.splitext(dsfname)
    if not os.path.isdir("models"):
        os.path.mkdir("models")
    outfile = os.path.join("models", f'{dsname}-{args.nestimators}-{args.max_depth}.pkl')
    file = np.load(ds, allow_pickle=True)
    X = file['data']
    y = file['labels'][:,0]
    if y[0] == -1:
        print(f"The dataset {dsname} does not contain labels. Training cannot proceed.")
        exit()
    print("initial:", X.shape, y.shape)
    if args.filter_na:
        print("Filtering class N/A.")
        flt = y != 18
        X = X[flt]
        y = y[flt]
        print("after filtering:", X.shape, y.shape)
        outfile = os.path.join("models", f'{dsname}-filter-{args.nestimators}-{args.max_depth}.pkl')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    print("train:", X_train.shape, y_train.shape, np.unique(y_train, return_counts=True)[1]/X_train.shape[0])
    print("val:", X_val.shape, y_val.shape, np.unique(y_val, return_counts=True)[1]/X_val.shape[0])
    print("test:", X_test.shape, y_test.shape, np.unique(y_test, return_counts=True)[1]/X_test.shape[0])

    nclasses = y.max() + 1

    data = (X_train, y_train, X_val, y_val, X_test, y_test)

    xgb1 = XGBClassifier(learning_rate=0.1,
                    n_estimators=args.nestimators,
                    max_depth=args.max_depth,
                    min_child_weight=1,
                    reg_alpha=0.01,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='multi:softmax',
                    nthread=args.nthreads,
                    num_class=nclasses,
                    seed=27,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    tree_method='gpu_hist',
                    gpu_id=args.cuda_device)

    fitXgb(xgb1, *data, epochs=args.nepochs, outfile=outfile)
