# splash

Collection of scripts to detect the splashback radius in Galaxy clusters using weak lensing data.
The cosmology is hardcoded, see Contigiani+2018.

## Files

* cluster.py : where the classes cluster_sample and cluster are defined.

* profile.py : where lensing profiles are computed: NFW, DK14.

* cosmic_noise.py : where the machinery required for the computation of cosmic noise is located.

## Directories

* Contigiani2018 : notebooks and scripts used for Contigiani, Hoekstra & Bahe 2018.

## Documentation

You can access every method's docstring by using the help() function in python.

## References

* What is splashback? Benedikt Diemer and Andrey V. Kravtsov, 2014, APJ ([arxiv](https://arxiv.org/abs/1401.1216))

* Measurement: E. Baxter, C. Chang, B. Jain et al., 2017, MNRAS ([arxiv](http://arxiv.org/abs/1702.01722))

* Everything here is expected to work on the data presented in Henk Hoekstra, Ricardo Herbonnet, Adam Muzzin, Arif Babul, Andi Mahdavi, Massimo Viola, Marcello Cacciato, 2015, MNRAS ([arxiv](https://arxiv.org/abs/1502.01883))

<!-- * Contigiani2018 is for the temp work. -->
