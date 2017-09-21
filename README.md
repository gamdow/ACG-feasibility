* Clone this repository
* Change directory: `$ cd AGC-ACG-feasibility`
* Build the container: `$ sudo docker build -t support .`
* Start the container with port forwarding and mount curent directory as `/root/working`: `$ sudo docker run -p 8888:8888 -v "$(pwd)":/root/working/ -it support`
* Access the following url in a browser: http://localhost:8888
