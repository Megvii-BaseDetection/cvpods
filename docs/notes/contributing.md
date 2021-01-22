# Contributing to cvpods
We want to make contributing to this project as easy and transparent as
possible.

## Issues
We use Gitlab issues to track public bugs and questions.
Please make sure to follow one of the
[issue templates](https://git-core.megvii-inc.com/zhubenjin/cvpods/issues)
when reporting any issues.

## Merge Requests
We actively welcome your mr(merge requests).

However, if you're adding any significant features, please
make sure to have a corresponding issue to discuss your motivation and proposals,
before sending a PR. We do not always accept new features, and we take the following
factors into consideration:

1. Whether the same feature can be achieved without modifying cvpods.
cvpods is designed so that you can implement many extensions from the outside, e.g.
those in [playground](https://git-core.megvii-inc.com/zhubenjin/cvpods_playground).
If some part is not as extensible, you can also bring up the issue to make it more extensible.
2. Whether the feature is potentially useful to a large audience, or only to a small portion of users.
3. Whether the proposed solution has a good design / interface.
4. Whether the proposed solution adds extra mental/practical overhead to users who don't
   need such feature.
5. Whether the proposed solution breaks existing APIs.

When sending a MR, please do:

1. create your branch from branch `megvii`.
2. If you've added code that should be tested, add tests.
3. If APIs are changed, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If a PR contains multiple orthogonal changes, split it to several PRs.
7. If you haven't already, complete the Contributor License Agreement ("CLA").

## License
By contributing to cvpods, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
