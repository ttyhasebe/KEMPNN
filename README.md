# KEMPNN (Knowledge-Embedded Message-Passing Neural Networks)

KEMPNN (Knowledge-Embedded Message-Passing Neural Networks) is a message-passing neural networks that
can be supervied togather with human-annotated knowledge annotations on the molecular graphs to improve the accuracy.

This package is an implementation of the paper: [Knowledge-Embedded Message-Passing Neural Networks: Improving Molecular Property Prediction with Human Knowledge](https://doi.org/10.1021/acsomega.1c03839) by the author.


## Getting started

To use this package, please clone this repository to your local disk.

You can also install KEMPNN as a package with dependencies if needed (torch is not included in dependencies) 
after cloning the repository:

    $ pip install -e .

### Requirements
This program is tested on the following environments:

* Python 3.9, torch 1.9.1, cuda11.1
* Python 3.8, torch 1.7.1, cuda10.1


### Examples

* To use the KEMPNN with custom dataset, knowledge, and training parameters, please refer to the example program [/examples/custom_dataset.py](/examples/custom_dataset.py).
* To gain more control over model and training using torch models, optimizer, dataloader etc., please refer to the example program [/examples/custom_training.py](/examples/custom_training.py).

### Training and Evaluation from CLI

By executing kempnn.py (on the project root), you can use KEMPNN through command-line interface.
In the CLI, the KEMPNN is trained with hyperparameter optimization and evaluated in multiple runs.

The following command will train KEMPNN with knowledge supervision enabled, frac. train=0.8 on ESOL dataset:

    $ python kempnn.py ESOL --frac_train 0.8 --save

or execute just a single run for testing with --single option without saving weights:

    $ python kempnn.py ESOL --frac_train 0.8 --single

To train standard MPNN with frac. train=0.4 on Lipophilicity dataset,

    $ python kempnn.py Lipop --frac_train 0.4 --no_knowledge --save

To train KEMPNN with no set2set aggregation and frac. train=0.2 on Tg dataset

    $ python kempnn.py PolymerTg --frac_train 0.2 --no_set2set --save

To see the detailed description of this CLI:

    $ python kempnn.py -h


### CLI Results

The above commands will report the result on console and
output XXX_eval.json and XXX_opt.json if the "--save" option is enabled,
which reports model performance (RMSE) and hyperparameter optimization results.

Example of the output:

    {
        "test_rmse": [
            "63.4952", // Mean RMSE on test set
            "0.0"    // Standard deviation of RMSE on test set
        ],
        "val_rmse": [
            "68.14108",   // Mean RMSE on validation set
            "0.0" // Standard deviation of RMSE on validation set
        ],
        // The following is the RMSE at the epoch of lowest validation error. These values are not used in paper.
        "best_test_rmse": [
            "66.53468",
            "0.0"
        ],
        "best_val_rmse": [
            "64.82107",
            "0.0"
        ],
        // The following are the results for each evaluation run.
        "results": [...],
        ...
    }



## Cite

If you use KEMPNN for your research or wish to refer to the baseline results, please cite the following paper:

    @article{hasebe2021kempnn,
      title={Knowledge-Embedded Message-Passing Neural Networks: Improving Molecular Property Prediction with Human Knowledge},
      author={Hasebe, Tatsuya},
      journal={ACS omega},
      volume={6},
      number={42},
      pages={27955--27967},
      year={2021},
      publisher={ACS Publications}
    }
