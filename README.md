* Clone this repository: `$ git clone https://github.com/gamdow/ACG-feasibility.git`
* Change directory: `$ cd ACG-feasibility`
* Build the container: `$ sudo docker build -t support .`
* Start the container with port forwarding and mount current directory as `/root/working`: `$ sudo docker run -p 8888:8888 -v "$(pwd)":/root/working/ -it support`
* Access the following url in a browser: http://localhost:8888
