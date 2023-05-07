In this code, we convert the `PartNet-Mobility` data and the `PartNet` data predicted by our model into the input format of `ANCSH`, and then train `ANCSH` and `ANCSH (semi-weakly)` which proposed in our paper.


### Convert data

First run the following script to convert the `mobility_v2.json` format file into `shape2motion` format.

```shell script
sh convert.sh
```

### Train baseline ANCSH

Render the input data of `ANCSH` on the `PartNet-Mobility` dataset

```shell script
cd articulated-pose && sh gendata_partnet_mobility.sh
```

Train the `ANCSH` baseline

```shell script
cd articulated-pose && sh ancsh.sh
```

### Train our ANCSH(semi-weakly)

Render the input data on the `PartNet` dataset, and expand the training data.

```shell script
cd articulated-pose && sh gendata_partnet.sh
```

Before training our `ANCSH(semi-weakly)`, please comment the `exp`, `baseline` and `joint_baseline` of `ANCSH` in `articulated-pose/global_info.py`, and uncomment those of `ANCSH(semi-weakly)`. Then run
the following script to train our `ANCSH(semi-weakly)`.

```shell script
cd articulated-pose && sh ancsh_semi_weakly.sh
```
