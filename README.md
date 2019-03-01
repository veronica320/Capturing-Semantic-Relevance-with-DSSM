# Summary

This is the project for FUTURE CAMP 2018. With a Keras re-implementation of DSSM/CDSSM, the model captures semantic relevance between texts.

# Code Structure

* conf.py: configuration of parameters and directory names.
* data_provider.py: data providing utility functions.
* model_structure: the nn structure for DSSM/CDSSM/CGRU.
* train.py: training the model.
* eval.py: evaluating the model.

# How to run the code

1. Set relevant parameters according to your choice in conf.py.
2. Run train.py.
3. Run eval.py.

## References

* Huang, P. S., He, X., Gao, J., Deng, L., Acero, A., & Heck, L. (2013, October). Learning deep structured semantic models for web search using clickthrough data. In Proceedings of the 22nd ACM international conference on Conference on information & knowledge management (pp. 2333-2338). ACM.

* Shen, Y., He, X., Gao, J., Deng, L., & Mesnil, G. (2014, April). Learning semantic representations using convolutional neural networks for web search. In Proceedings of the 23rd International Conference on World Wide Web (pp. 373-374). ACM.

* Shen, Y., He, X., Gao, J., Deng, L., & Mesnil, G. (2014, November). A latent semantic model with convolutional-pooling structure for information retrieval. In Proceedings of the 23rd ACM International Conference on Conference on Information and Knowledge Management (pp. 101-110). ACM.

