#  Live Neuro Verification & Validation Checklist

##  1. SRS Verification

| Item | Criteria                                                                 | Checked |
|------|--------------------------------------------------------------------------|---------|
| 1.1  | All functional and non-functional requirements are listed                | [ ]     |
| 1.2  | Requirements are feasible and implementable                              | [ ]     |
| 1.3  | Each requirement is traceable to project objectives                      | [ ]     |
| 1.4  | Feedback is reviewed and resolved on GitHub                              | [ ]     |

---

##  2. Design Verification

| Item | Criteria                                                                 | Checked |
|------|--------------------------------------------------------------------------|---------|
| 2.1  | Unit tests cover all critical paths and edge cases                       | [ ]     |
| 2.2  | Integration tests confirm inter-module communication                     | [ ]     |
| 2.3  | Code walkthroughs were conducted with peers/supervisor                   | [ ]     |
| 2.4  | Code conforms to architecture described in the design document           | [ ]     |
| 2.5  | Doxygen diagrams accurately represent module dependencies                | [ ]     |

---

##  3. Implementation Verification

| Item | Criteria                                                                 | Checked |
|------|--------------------------------------------------------------------------|---------|
| 3.1  | Test coverage ≥ 90% (`pytest-cov`)                                       | [ ]     |
| 3.2  | Flake8 passes with no major issues                                       | [ ]     |
| 3.3  | Mypy passes with consistent type annotations                             | [ ]     |
| 3.4  | GitHub Actions successfully builds the software on all platforms         | [ ]     |
| 3.5  | Matplotlib `image_comparison` tests validate plotting correctness        | [ ]     |

---

## ️ 4. Edge Case Testing

| Item | Criteria                                                              | Checked |
|------|-----------------------------------------------------------------------|---------|
| 4.1  | System handles extreme neural values without visual errors            | [ ]     |
| 4.2  | Sparse data (e.g. NaNs, missing trials) does not break plotting logic | [ ]     |
| 4.3  | Corrupted/unusual formats trigger helpful error messages              | [ ]     |
| 4.4  | Axis scaling, zoom, and annotations work with outlier data            | [ ]     |


---

##  5. Usability Validation

| Item | Criteria                                                                 | Checked |
|------|--------------------------------------------------------------------------|---------|
| 5.1  | All participants can complete key tasks without external help            | [ ]     |
| 5.2  | Time to first plot is recorded for all participants                      | [ ]     |
| 5.3  | Survey responses are collected and analyzed                              | [ ]     |
| 5.4  | Interface revisions are made based on usability trends                   | [ ]     |

---

##  6. Portability and Reusability

| Item | Criteria                                                                 | Checked |
|------|--------------------------------------------------------------------------|---------|
| 6.1  | Software installs and runs correctly on Linux, Windows, and macOS        | [ ]     |
| 6.2  | Platform-specific bugs are documented and addressed                      | [ ]     |
| 6.3  | Code reviewers confirm that major modules are logically separated        | [ ]     |
| 6.4  | Review checklist for modularity is completed and issues logged on GitHub | [ ]     |
