### Organizational Mining: Process Mining on organizational perspective
*Organizational mining* focuses on discovering organizational knowledge such as organizational structures and social networks \[1\]. For example, *organizational model mining* can help extract the structural patterns of employees, and may support decision-makers in better understanding *de facto* grouping of human resources related to a business process in an organization \[2\].

Whereas lion's share of research in process mining has been devoted to the control-flow perspective of business processes, event log data may also contain a wealth of information related to other pespectives, e.g. organizational perspective \[3\]. 

The research topic of organizational mining was first proposed by Song and van der Aalst \[1\]. Three types of mining are discussed:
* *Discovery*: deriving models to describe the reality as reflected by event logs, e.g. organizational models, social networks, assignment rules, resource behavior.
* *Conformance*: investigating the (mis-)alignment between modeled behavior and those observed in event logs.
* *Enhancement*: enriching or improving existing models using information extracted from event logs.

My current research aims at gaining a more comprehensive understanding of organizational mining, with a specific focus on the topic of organizational model mining for which the object of study is the grouping of resources. The current project, *OrgMiner*, is a Python library developed for research on the topic of organizational mining. 

### The OrgMiner library
*OrgMiner* is based on our recent research progress reported in a preprint paper submitted to the journal of Information Systems \[5\]. The library includes all the necessary modules and functions to implement the approach proposed in the paper and thus enables replicating the designed experiments for evaluation.

Furthermore, this library is established such that it allows a user to build their own approaches for organizational model discovery and conformance checking, either by adjusting and extending the existed methods, or by inventing new methods/modules that extend the library.

### How to use
#### Prerequisite
OrgMiner is built using Python ([What is Python?](https://www.python.org/)) hence it is a prerequisite to have Python installed on your machine. 

We recommend using [Anaconda Python] (https://www.anaconda.com/distribution/), a distribution of Python with enhanced support of package management which will make life easier especially in resolving dependencies. You may choose to install the minimal core of Anaconda Python only: [Miniconda] (https://docs.conda.io/en/latest/miniconda.html).

#### Install OrgMiner
OrgMiner can be installed by the Anaconda Python package manager. In Unix systems,

In Windows,


#### Replicate experiments in the paper \[5\]


#### Make use of OrgMiner
OrgMiner is developed as a library meaning users are more than welcome to build their own main programs by importing the modules/methods from OrgMiner. Examples can be found as the main files that replicate the experiments, and also the 2 linked repositories related to 2 publications.

See the Documentation section for more information for an instruction on how to use the library for building a program and/or extending the library as a contributor.

### Documentation (WIP)
We are working on a structured documentation for OrgMiner library, which will be available through Read the Docs soon.

### Linked repositories
* [Project Arya](https://github.com/roy-jingyang/Arya) provides an interactive interface as a simple client-server application. It can be seen as an alternative to some of the CLI (command-line interface) programs in this repository.
* [BPM 2018 paper prototype](https://github.com/roy-jingyang/bpm-2018-Yang_Find) presents a demo program that implements the approach proposed in the paper \[2\], which is built upon an early version of OrgMiner. Users are advised to visit the updated version of this prototype which can be found under the main folder in this repository (`main/archived/bpm2018yang`).
* [CAiSE 2020 paper prototype](https://github.com/roy-jingyang/caise-2020-Ouyang_Discovering) presents a demo program that implements the approach proposed in the paper \[4\], which is built upon an early version of OrgMiner. Users are advised to visit the updated version of this prototype which can be found under the main folder in this repository (`main/archived/caise2020ouyang`).
* [PM4Py](http://pm4py.org/) is a general purpose process mining library in Python. Some of the features in OrgMiner is built using modules from PM4Py.

#### References:
###### 1. Song, M., & van der Aalst, W. M. P. (2008). Towards comprehensive support for organizational mining. Decision Support Systems, 46(1), 300–317.
###### 2. Yang, J., Ouyang, C., Pan, M., Yu, Y., & ter Hofstede, A. H. M. (2018). Finding the “Liberos”: Discover Organizational Models with Overlaps. In M. Weske, M. Montali, I. Weber, & J. vom Brocke (Eds.), Business Process Management (pp. 339–355). Cham: Springer International Publishing.
###### 3. Van der Aalst, W. M. P. (2016). Process Mining: Data Science in Action (2nd ed.). Springer.
###### 4. Ouyang, C., Leyer, M., Yang, J., Sindhgatta, R. (2020). Discovering Organizational Resource Grouping Behavior from Event Logs. Manuscript submitted to 32nd International Conference on Advanced Information Systems Engineering (CAiSE 2020), under review.
###### 5. (Citation to the IS paper to be added.)

