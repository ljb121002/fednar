## Code to reproduce our paper: FedNAR: Federated Optimization with Normalized Annealing Regularization

### Requirements
```shell
pip install -r requirements
```

### Training
To access all the training scripts, you can find them in the `./scripts` directory. For all the scripts, you can enable `--use_nar` to use FedNAR variants. For the Shakespeare dataset, please generate the required data by following the instructions provided in `./LEAF/shakespeare`.

### Acknowledgements
We would like to extend our gratitude to the authors of the following works: [FedExP](https://github.com/divyansh03/fedexp), [FedDyn](https://github.com/alpemreacar/FedDyn), and [LEAF](https://github.com/TalwalkarLab/leaf). Our codes are built upon their open-source projects.