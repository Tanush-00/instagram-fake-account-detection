# InstaFake dataset (fake account part, version fake-v1.0)

`fakeAccountData.json` and `realAccountData.json` are copied **unchanged** from
[github.com/fcakyon/instafake-dataset](https://github.com/fcakyon/instafake-dataset), folder `data/fake-v1.0`.

- **Authors:** Fatih Cagatay Akyon and Esat Kalfaoglu
- **Paper:** F. C. Akyon and M. E. Kalfaoglu, *Instagram Fake and Automated Account Detection*, 2019. [arXiv:1910.03090](https://arxiv.org/abs/1910.03090)
- **License:** Creative Commons Attribution-NonCommercial 4.0 International ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/))
- **Contents:** 200 fake and 994 genuine Instagram accounts, anonymised by the authors (no usernames). Each account has 8 profile features and a label:

| Field | Meaning |
|---|---|
| `userMediaCount` | number of posts |
| `userFollowerCount` | number of followers |
| `userFollowingCount` | number of accounts it follows |
| `userBiographyLength` | number of characters in the bio |
| `userHasProfilPic` | 1 if it has a profile picture |
| `userIsPrivate` | 1 if the account is private |
| `usernameDigitCount` | number of digits in the username |
| `usernameLength` | number of characters in the username |
| `isFake` | 1 = fake, 0 = genuine |

The notebooks remove 29 repeated rows after loading. The files themselves are not modified.
