FROM ros:noetic-ros-base-focal

RUN apt-get -y update 
RUN apt-get -y upgrade
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

RUN apt-get install -yq curl build-essential
RUN apt-get install -yq python3-pip
RUN apt-get install -y git
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install -yq ros-noetic-cv-bridge
RUN apt-get install -yq ros-noetic-rviz
RUN apt-get -y update 
RUN apt-get install -yq ros-noetic-libfranka ros-noetic-franka-ros
RUN apt-get install -yq wget
RUN mkdir /home/workspace

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install CoTracker
RUN git clone https://github.com/facebookresearch/co-tracker /home/co-tracker
RUN pip install -e /home/co-tracker/
RUN pip install matplotlib flow_vis tqdm tensorboard imageio[ffmpeg]
RUN mkdir /home/co-tracker/checkpoints
RUN wget -P /home/co-tracker/checkpoints https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth

WORKDIR /home/workspace