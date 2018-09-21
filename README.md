## Organizational Mining: Process Mining on organizational perspective

Whereas lion's share of research in process mining has been devoted to the control-flow perspective of business processes, event log data may also contain a wealth of information related to other pespectives, e.g. organizational perspective, time perspective \[1\]. 

*Organizational mining* focuses on discovering organizational knowledge such as organizational structures and social networks \[2\]. For example, *organizational model mining* can support decision-makers to better understand the *de facto* grouping of human resources related to a business process \[3\].

The organizational mining research topic was first proposed by Song and van der Aalst \[2\], in which three types of mining are discussed:
* *Discovery*: construct models that reflect the reality. Topics of interest include:
    * organizational model mining: find the grouping of resources along with the relationships between resource groups and tasks.
    * social networks: use Social Network Analysis \(SNA\) techniques to help understand the structure of communication between resources and groups.
    * rules: discover rules related to staff assignment and \(runtime\) activity distribution.
* *Conformance (Alignment)*: examine whether the modeled behaviour matches the observed.
* *Extension*: enrich an existing model by extending it through the projection of information extracted from the logs onto the initial model.

My current research aims at gaining a more comprehensive understanding of organizational mining, and project *ProMorg* is a prototype implementing the tools and techniques related to this research topic.

##### References:
###### 1. Van der Aalst, W. M. P. (2016). Process Mining: Data Science in Action (2nd ed.). Springer.
###### 2. Song, M., & van der Aalst, W. M. P. (2008). Towards comprehensive support for organizational mining. Decision Support Systems, 46(1), 300–317.
###### 3. Yang, J., Ouyang, C., Pan, M., Yu, Y., & ter Hofstede, A. H. M. (2018). Finding the “Liberos”: Discover Organizational Models with Overlaps. In M. Weske, M. Montali, I. Weber, & J. vom Brocke (Eds.), Business Process Management (pp. 339–355). Cham: Springer International Publishing.
