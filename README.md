# Shift happens: Crowdsourcing metrics and test datasets beyond ImageNet

*ICML 2022 workshop proposal (under review)*

While the popularity of robustness benchmarks and new test datasets increased over the past years, the performance of computer vision models is still largely evaluated on ImageNet directly, or on simulated or isolated distribution shifts like in ImageNet-C.
The goal of this two-stage workshop is twofold:
First, we aim to enhance the landscape of robustness evaluation datasets for computer vision and devise new test settings and metrics for quantifying desirable properties of computer vision models.
Second, we expect that these improvements in the model evaluation lead to a better guided and, thus, more efficient phase for the development of new models. This incentivizes development of models and inference methods with meaningful improvements over existing approaches  with respect to a broad scope of desirable properties.
Our goal is to bring the robustness, domain adaptation, and out-of-distribution detection communities together to work on a new broad-scale benchmark that tests diverse aspects of current computer vision models and guides the way towards the next generation of models.


| ![overview.svg](overview.svg) | 
|:--:| 
| *Illustration of the envisioned procedure and outcome of this workshop: We will crowdsource and curate a collection of tasks and corresponding datasets highlighting interesting aspects of ImageNet-scale models. A set of reference models will be evaluated on these datasets during the benchmark, yielding an initial set of scorecards for commonly used ImageNet models. Following the benchmark creation, more models and new techniques can be evaluated, enabling a more holistic view on the performance of practically relevant computer vision models.* |

## Submissions & Information

The benchmark will be available on [github.com/shift-happens-benchmark/icml-2022](https://github.com/shift-happens-benchmark/icml-2022).
API docs are available on [shift-happens-benchmark.github.io/icml-2022/](https://shift-happens-benchmark.github.io/icml-2022/).

The workshop aims to build up a range of evaluation datasets that
together allow for a detailed overview of a model's strengths and
weaknesses across a variety of tasks. The workshop will result in a
software package of datasets and benchmarks interesting to a large
community dealing with ImageNet size models, including practitioners
interested in seeing practically relevant properties and trade-offs
between models.

#### Submission types

Besides compatibility to ImageNet scale models, the scope of possible
benchmarks and datasets is intentionally broad: First, we encourage
submissions that provide their own evaluation criterion and discuss its
value in applications. Submissions should explain why the submitted
dataset and metric are well-suited to inform about the specified
property. Opening the benchmark to this form of submissions makes it
possible for communities interested in problems besides standard,
"accuracy-focused" settings. Second, we encourage submissions of
datasets that can be evaluated with standard criteria. This form of
submission imposes a low bar on developing new code contributions and
makes it possible to contribute in the form of well-recorded datasets.
Third, in both cases, it is possible to re-submit existing and
potentially published benchmarks, datasets, and evaluation settings,
known only in a particular community and make these datasets available
to a broader audience as part of a curated benchmark package. Examples
include small datasets that test an interesting distribution shift, such
as shifts occurring due to applications in the real world.

Within these three submission types, the design of the benchmark will
focus in particular on datasets falling into one or more of categories
below:

1.  Robustness to domain shifts (classification accuracy): A labeled
    dataset where the labels are (a subset of) the 1000 labels of
    ImageNet-2012. Optionally, model calibration, uncertainty, or open
    set adaptation can be tested. We especially encourage submissions
    focusing on practically relevant distribution shifts.

2.  Out-of-distribution detection: A labeled or unlabeled dataset of
    images that do not contain objects from any of the 1000
    ImageNet-2012 classes.

3.  New robustness datasets: Beyond the standard robustness evaluation
    settings (with covariate shift, label shift, ...), the workshop
    format enables submission of datasets that evaluate non-standard
    metrics such as the consistency of predictions, influence of
    spurious correlations in the dataset.

We ensure standardization of submitted datasets and evaluations
algorithms by providing a reference implementation with pre-defined
interfaces. These interfaces allow writing datasets and benchmarks that
are guaranteed to be compatible with a broad class of models. A critical
decision is to limit submissions to models compatible with ImageNet
pre-training: Given a batch of images, models will provide (at least)
class predictions and optionally features, class confidences, and an OOD
score. Given this information, each benchmark needs to define the
necessary mechanisms for evaluating and returning scores. Our reference
implementation (which will be extended in the coming weeks) is available
at <https://github.com/shift-happens-benchmark/iclr-2022>.

Submissions will be allowed to contain multiple related datasets, e.g.,
a dataset like ImageNet-C could have been submitted as a collection of
15 evaluation datasets, corresponding to the different corruptions
ImageNet-C is comprised of.

#### Evaluation criteria

Successful and accepted submissions to the workshop will contain a
technical report about the dataset design and data collection procedure
along with a code submission (under a suitable open-source license) that
will be added to the benchmark. After our workshop, this will ensure
that all benchmarks are accessible to the community.

Submissions will be judged according to the following criteria:

1.  Correctness: For labeled datasets, the labels should make sense to a
    human reviewer. For OOD datasets, no in-distribution objects can be
    visible on the images. During the review of large datasets, random
    samples and the worst mistakes of some models will be checked. The
    correctness will mainly be reviewed based on the submitted dataset
    (raw images), and the technical report. The main goal is to exclude
    "edge cases" and low-effort submissions from the benchmark that are
    not interesting to the community, and might be hard to spot by
    automated evaluation.

2.  Novelty: Datasets which allow for a more insightful evaluation
    beyond the standard test accuracy of ImageNet are encouraged. This
    will be formally benchmarked by evaluating a set of standard models
    on the provided dataset. In the initial package, we will include (1)
    a set of (robustified) ResNet models, (2) models that provide an
    explicit OOD detection score, as well as (3) recent test-time
    adaptation methods. Evaluation should be done by the authors and
    included in their technical report. Models for evaluating a dataset
    will become part of the provided reference implementation. For
    accepted benchmarks, we will verify the author's results before
    posting the results on the public leaderboard.

3.  Difficulty for current models: If the task can easily be solved by
    humans but some models fail moderately or spectacularly, it is an
    interesting addition to the benchmark. As with the Novelty
    criterion, we expect authors to evaluate this score based on the
    provided reference implementation.

Besides the robustness and out-of-distribution detection communities
directly addressed by the default benchmark items mentioned above, this
workshop pre-eminently is meant to bring together different communities
that can contribute assets in the form of datasets and interesting
evaluation tasks. For example, researchers who work primarily on
modeling 3D objects might provide an interesting puzzle piece to be
integrated in a comprehensive evaluation suite.

During the workshop, we will encourage discussion on (1) model
properties that are often overlooked when evaluating machine learning
models and should be included in a comprehensive benchmark, on (2)
important practical properties of evaluation datasets and criteria, and
on (3) currently unavailable evaluations that would be desirable to be
developed in the future. Furthermore, we will host an online forum in
the period between the camera-ready deadline and the workshop to
facilitate constructive discussions about the accepted datasets.

## Submission Procedure and Reviewing

To meet the goals outlined above, we will organize a review process that
places a strong focus on the quality of the submitted datasets and
metrics and their implementation.

Besides encouraging community building around the benchmark, the
proposed review process will also be an experiment for implementing *a
review process that centers around the code submission*. We think that
including the community in an open review process will be an opportunity
to increase chances for later adaptations of the benchmark. Tools
developed for setting up this review process will later be released as
open-source tools.

In more detail, reviewing will be done in the following stages:

1.  Submission of a short, 2--4 page technical report on the dataset,
    including a description of how images were acquired, which
    evaluation metrics will be used, usage information, and plans to
    make the dataset accessible. The technical report should include
    reference results from running the provided models on the new
    dataset, and optionally additional experiments. Submissions must
    include a link to the dataset (hosted on a suitable platform), as
    well as code (building on top of the provided reference
    implementation) for implementing the evaluation process. Submissions
    will be coordinated on OpenReview, and reviewing is double-blind.

2.  As preparation for the review stage, all anonymized submissions will
    be public on OpenReview. In addition, we will create (anonymized)
    pull requests on the benchmark repository based on the submissions.
    Authors are responsible for preparing their submissions accordingly,
    and documentation for doing this correctly (and testing the
    submission prior to uploading on OpenReview) will be made available
    on the workshop page.

3.  In the reviewing phase, reviewers will judge the quality of both
    technical reports (on OpenReview) and submitted code (on GitHub),
    according to the criteria introduced above. In parallel to the
    reviewing phase, we will start running tests on the submitted
    benchmarks for a collection of established vision models. While
    adding comments on OpenReview will be limited to the reviewers, code
    review (and proposal of improvements) on GitHub is open to the
    public --- this also includes criticism of the data collection
    process described in the technical report. Our rationale is to limit
    OpenReview comments to a limited number of "formal" reviews. At the
    same time public discussion --- and community building relevant for
    the benchmark after the workshop ends --- will be encouraged on
    GitHub.

4.  In the discussion phase, authors are allowed to update both their
    technical report and the submitted code.

5.  After the final decisions, all submissions will be de-anonymized
    both on OpenReview and on GitHub. The outlined review process will
    ensure that for this final set of camera-ready submissions, a set of
    datasets with reviewed descriptions (submitted reports), and
    high-quality code ready to merge into the benchmark will be
    available. After the camera-ready phase, and after ensuring
    technical soundness of the submitted PRs, we will release a first
    version of the benchmark that is already suitable for contributing
    additional models, and techniques, as well as making suggestions on
    improving the benchmarks and metrics.

6.  Two weeks prior to the workshop, we will host a "hackathon" aimed at
    community building around the benchmark. For this, discussions will
    happen on GitHub, and the community will be able to contribute
    changes to the benchmark. The best contributions from this phase
    will get a short talk (time depends on the number of contributions)
    at the workshop.

We should note that we will make submission of code for review as easy
and convenient as possible for the authors: For example, the reference
package will make it possible to submit benchmark datasets with standard
metrics (e.g., accuracy on a new dataset), with a minimal code
submission, using helper functions already provided in the package.

## Related Software Packages

Below we list a set of related works in open-source software, benchmarks
and datasets released in the past years and gained popularity in
different communities. While some datasets are orthogonal to our effort,
we plan to seek active collaborations and discussions in case of
potential synergies. The organizing committee and invited speakers
already cover a considerable number of packages mentioned below.

#### [WILDS Benchmark](https://wilds.stanford.edu/)

WILDS is "a benchmark of in-the-wild distribution shifts spanning
diverse data modalities and applications, from tumor identification to
wildlife monitoring to poverty mapping". In contrast to the ShiftHappens
benchmark, WILDS is not primarily focused on the evaluation of
pre-trained ImageNet trained models but mainly considers the setting of
domain generalization on a broader range of tasks, which requires model
training.

However, we think that many synergies exist between our workshop goal
and the WILDS benchmark and are in contact with some of the authors who
will join the workshop as confirmed speakers.

#### [Robusta](https://github.com/bethgelab/robustness)

Robusta is a growing collection of helper functions, tools and methods for robustness evaluation
and adaptation of ImageNet scale models. The focus is on simple methods that work at scale.

#### [Visual Decathlon](https://www.robots.ox.ac.uk/~vgg/decathlon/)

The Visual Decathlon challenge requires simultaneously solving ten image
classification problems representative of very different visual domains.
For this challenge, the participants were allowed to use the train and
validation splits of the different datasets to train their classifier
(or several classifiers). While the Visual Declathon requires a training
phase, the envisioned ShiftHappens benchmark focuses on evaluating
ImageNet pre-trained models.

#### [Visual Task Adaptation Benchmark (VTAB)](https://github.com/google-research/task_adaptation)

VTAB contains 19 challenging downstream tasks for evaluating vision
models. The tasks stem from different domains such as natural images,
artificial environments (structured), and images captured with
non-standard cameras (specialized). VTAB focuses on task adaptation,
needs a lot of compute for fine-tuning the models on the target tasks,
and is, therefore, orthogonal to our proposed benchmark (which will only
contain test datasets).

#### [ImageNet-C](https://github.com/hendrycks/robustness), [ImageNet-P](https://github.com/hendrycks/robustness), [ImageNet-R](https://github.com/hendrycks/imagenet-r), [ImageNet-A](https://github.com/hendrycks/natural-adv-examples), [ImageNet-O](https://github.com/hendrycks/natural-adv-examples)

ImageNet-C, -P, -R, -A, and -O are ImageNet-compatible datasets that are
highly relevant to the workshop, widely adopted in the community, and
will be included as reference datasets into the benchmark (all of them
are published under suitable open-source licenses). We invited Thomas G.
Dietterich, one of the ImageNet-C authors to share his thoughts about
his efforts in robustness evaluation during the workshop.

#### [ObjectNet](https://objectnet.dev/)

Similar to ImageNet-C and variants, ObjectNet is a currently isolated
benchmark dataset that fits the workshop's scope. We will explore
possibilities for including ObjectNet in the reference implementation
--- due to the special license of ObjectNet; this will require
additional attention from the ObjectNet authors.

#### [Model vs. Human](https://github.com/bethgelab/model-vs-human)

The `modelvshuman` package is centered around benchmarking the
similarity between ImageNet trained models and human subjects while
solving the same task. Co-organizers Wieland B. and Matthias B. are
actively involved in this project, and we are discussing possibilities
of leveraging synergies between this package and the ShiftHappens
benchmark.

#### [RobustBench](https://github.com/RobustBench/robustbench)

The `robustbench` package, initiated by co-organizer Matthias H.,
focuses on evaluating robustness against adversarial perturbations by
combining different state-of-the-art attack techniques. It also features
a leaderboard for robustness against the common corruptions in
CIFAR-C/Imagenet-C. As with other related packages, we will explore
synergies and potentially leverage functionality from the robustbench
package in our reference implementation.

#### [Foolbox](https://github.com/bethgelab/foolbox)

The `foolbox` package is a popular package around benchmarking
adversarial robustness. Co-organizer Wieland B. is one of the initiators
of this package. While we do not plan to focus specifically on
robustness to adversarial examples, we do not exclude the possibility
that some submissions make use of `foolbox` or related libraries for
robustness evaluation. However, an important criterion will be that
these datasets test practically relevant aspects (e.g., adding
adversarial patches to an image or other practically conceivable
scenarios).

#### [Timm](https://github.com/rwightman/pytorch-image-models)

The `timm` package is an increasingly popular package (>14,000 GitHub
stars) for state-of-the-art computer vision models trained with PyTorch.
The package includes reference results for robustness on
ImageNet-A,-R,-C[^1], and we will explore possibilities of leveraging
the well-designed API and variety of models for our benchmark, e.g. by
making it easier to include `timm` models in the evaluation and
generation of model scorecards.

[^1]: See
    <https://github.com/rwightman/pytorch-image-models/tree/master/results>
