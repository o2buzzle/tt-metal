# Getting up and running

We've decided to put all of the information for this repo into our official
documentation, including info to get off the ground and started. Please read
and perform the following to

- set up your machine
- build the docs
- view the docs

1. Log into your machine. You should have been provided an IP and temporary
username. Remember to connect to the Tenstorrent Cloud VPN if you were
instructed to.

Proceed to step 2 below if you're not a developer.

You must create your own user account with sudo access on this machine. Please
do the following and create a password. This will be the account you're using:

```
sudo adduser <USERNAME>
sudo usermod -aG sudo <USERNAME>
```

Then log out and log back in under that new ``<USERNAME>``.

2. Create an SSH key for this machine.

```
ssh-keygen
```

You can keep everything default, so press `<ENTER>` until it finishes.

3. Add the key to your Github profile. Please refer to [SSH keys on
   Github](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

4. Clone this repo and navigate to its folder.
```
git clone git@<HOST>:<REPO>.git --recurse-submodules
cd <REPO>
```

5. Build and activate the TT-Metal environment:
```
source ./build_tt_metal.sh
source build/python_env/bin/activate
```

6. Now build HTML pages for the docs and launch a
   web server on the port `<port>`.

```
cd docs
PORT=<port> make all
```

7. Navigate to the docs page.

Navigate your web browser to `http://<ip address>:<port>`, where `<ip address>`
is the IP address of the machine on which you launched the web server. For
example: `http://10.250.37.37:4242`, for port ``4242``.

8. Open a new terminal window and click the very first `Getting Started` link.

9. Follow the instructions to set up the library.
