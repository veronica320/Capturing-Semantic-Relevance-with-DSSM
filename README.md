# Summary

This is the directory for FUTURE CAMP NLP WEEK2 project. With a Keras implementation of DSSM/CDSSM/CGRU, the model determines whether two given sentences are related to each other.

# Structure

* conf.py: configuration for parameters and directory names.
* model_structure: the nn structure for DSSM/CDSSM/CGRU.
* train.py: training model.
* eval.py: evaluating model.

# How to run the code

1. Set relevant parameters according to your choice in conf.py.
2. Run train.py.
3. Run eval.py.
* You need to put data files under directory camp_dataset2/ prior to running the code. The author is not authorized to publish the proprietory data, so you may have to obtain data yourself.

## References

* Shen, Y., He, X., Gao, J., Deng, L., & Mesnil, G. (2014, November). A latent semantic model with convolutional-pooling structure for information retrieval. In Proceedings of the 23rd ACM International Conference on Conference on Information and Knowledge Management (pp. 101-110). ACM.

* Huang, P. S., He, X., Gao, J., Deng, L., Acero, A., & Heck, L. (2013, October). Learning deep structured semantic models for web search using clickthrough data. In Proceedings of the 22nd ACM international conference on Conference on information & knowledge management (pp. 2333-2338). ACM.
