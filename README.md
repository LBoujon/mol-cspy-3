# mol-CSPy software package
This is the official repository for mol-CSPy, the Day group crystal structure prediction software developed in Southampton, UK.

<center>
<img src=logo.png width=500>
</center>

## Important !!!
Mol-CSPy has recently undergone a major change and `git pull` may no longer work for you.
If you have made any local changes to mol-CSPy you should make a backup of those, before moving forwards.
Before you next do `git pull`, do `git config pull.rebase true`.
Alternatively, clone a fresh copy of the repo.
If it is important to you to continue with mol-CSPy v2, you can find it on the `legacy` branch.

### Why has something changed?
The old version of mol-CSPy was created outside our typical git workflow. This yielded a divergent branch whose differences could not be consolidated with git.
This new verison is closely inline with our internal version of CSPy, which will make it easier to maintain the public branch, and deliver our latest feature developments.

We recommend mol-CSPy v3+ as a more stable and feature-complete alternative to prior versions.

## Documentation
For installation and general use information, our documentation is hosted on Read the Docs at <https://mol-cspy.readthedocs.io/en/latest> 

NOTE: Commands and python modules used for this software package have the prefix `cspy` for legacy reasons.

## Reporting bugs
Bugs and other issues should be reported in the issue tracker - please check if an bug/issue has already been reported by another user. When reporting a bug please consider the following:

- Which version of mol-CSPy are you using?
- Are you running mol-CSPy locally (on a personal computer) or on a computing cluster?
- Are there any clear steps to reproducing the bug?
- What is the expected outcome if the bug was not present?

## Contributing to mol-CSPy
mol-CSPy is an open source project that welcomes feedback and contributions from everyone.

If you wish to make a contribution, we would encourage users to follow these steps:

1. Fork the repository and clone it locally.
2. Create a new branch for your feature or bug fix: `git checkout -b feature-name` or `git checkout -b bug-fix`.
3. Make your changes and commit them: `git commit -am 'Add new feature'`.
4. Push to the branch: `git push origin feature/my-feature` or `git push origin bugfix/my-bug-fix`.
5. Submit a pull request.

Before submitting a pull request, please consider the following:

- Does your code adhere to the existing style of the project?
- Have you clearly noted the changes in the CHANGELOG?
- Have you made sure that there is only one updated feature per commit?

Finally, by contributing to mol-CSPy, you agree that your contributions will be licensed under the project's [LICENSE](LICENSE).

## License
mol-CSPy is licensed under the terms of the GNU General Public License version 3 (GPL-3.0). You can read the full license text in the [LICENSE](./LICENSE) file included in the repository.

## Citation
The use mol-CSPy for research or business should be cited with the following publications:
- Case, D. H., Campbell, J. E., Bygrave, P. J., & Day, G. M. (2016). Convergence Properties of Crystal Structure Prediction by Quasi-Random Sampling. Journal of Chemical Theory and Computation, 12(2), 910–924

For calculations employing the threshold algorithm, please cite the following publication:
- Yang, S., Day, G.M., Global analysis of the energy landscapes of molecular crystal structures by applying the threshold algorithm. Commun Chem 5, 86 (2022). https://doi.org/10.1038/s42004-022-00705-4
- Butler, P.W.V., Day, G.M., Reducing overprediction of molecular crystal structures via threshold clustering. PNAS, 120, 23 (2023). https://doi.org/10.1073/pnas.2300516120

For calculations employing the quasi-random basin hopping algorithm, please cite the following publication:
- Yang, S., Day, G.M., J. Chem. Theory Comput. 2021, 17, 3, 1988–1999

## Authors

This project is made possible by the following contributors:

- Graeme M. Day
- Sophie Bennett
- James Bramley
- Patrick W. V. Butler
- Peter J. Bygrave
- David H. Case
- Chi Y. Cheng
- Ramon Cuadrado
- Joshua Dickman
- Jordan Dorrell
- Lucia Gigli
- Joseph Glover
- Roohollah Hafizi
- Jay Johal
- Jennie Martin
- David P. McMahon
- Jonas Nyman
- Pedro Juan Royo
- Peter R. Spackman
- Christopher R. Taylor
- Jack Yang
- Shiyue Yang
- Zhengtao Zhou