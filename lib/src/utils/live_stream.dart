import 'package:drop_shadow/drop_shadow.dart';
import 'package:flutter/material.dart';
import 'package:flutter_vlc_player/flutter_vlc_player.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:railway/src/utils/constants.dart';

class LiveStream extends StatefulWidget {
  final String url;
  const LiveStream({super.key, required this.url});

  @override
  State<LiveStream> createState() => _LiveStreamState();
}

class _LiveStreamState extends State<LiveStream> {
  late VlcPlayerController _vlcPlayerController;
  late Future<void> _initializeVideoPlayerFuture;

  @override
  void initState() {
    super.initState();

    try{
      _vlcPlayerController = VlcPlayerController.network(
        widget.url,
        hwAcc: HwAcc.full,
        autoPlay: true,
        options: VlcPlayerOptions(),
      );
      _initializeVideoPlayerFuture = _vlcPlayerController.initialize();
    } catch(e){
      Fluttertoast.showToast(msg: "Unable to load Video");
    }

  }

  @override
  void dispose() {
    _vlcPlayerController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return DropShadow(
      blurRadius: 3,
      offset: const Offset(0, 3),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 500),
        curve: Curves.easeInOutQuint,

        decoration: BoxDecoration(
            border: Border.all(width: 1, color: Colors.black),
            borderRadius: BorderRadius.circular(11),
          color: Colors.black54
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(10),

          child: FutureBuilder(
            future: _initializeVideoPlayerFuture,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.done) {
                return VlcPlayer(
                  controller: _vlcPlayerController,
                  aspectRatio: 16/9,
                  placeholder: const Center(
                    child: CircularProgressIndicator(),
                  ),
                );
              } else {
                return const Center(child: CircularProgressIndicator());
              }
            },
          ),
        ),
      ),
    );
  }
}


// Class to play Recorded Video Clips
class PlayVideoClip extends StatefulWidget {
  final String url;
  const PlayVideoClip({super.key, required this.url});

  @override
  State<PlayVideoClip> createState() => _PlayVideoClipState();
}

class _PlayVideoClipState extends State<PlayVideoClip> {
  late VlcPlayerController _vlcPlayerController;
  late Future<void> _initializeVideoPlayerFuture;
  var marginValue = 5.0;
  var myOpacity = 0.75;
  Color buttonColor = secondaryColor;
  bool isPlaying = false;


  @override
  void initState() {
    super.initState();

    try{
      _vlcPlayerController = VlcPlayerController.network(
        widget.url,
        hwAcc: HwAcc.full,
        autoPlay: false,
        options: VlcPlayerOptions(),
      );
      _initializeVideoPlayerFuture = _vlcPlayerController.initialize();
    } catch(e){
      Fluttertoast.showToast(msg: "Unable to load Video");
    }

  }

  @override
  void dispose() {
    _vlcPlayerController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return DropShadow(
      blurRadius: 5,
      color: Colors.black54.withOpacity(0.5),
      offset: const Offset(0, 4),
      child: AnimatedOpacity(
        opacity: myOpacity,
        duration: const Duration(seconds: 2),
        curve: Curves.easeOutQuart,

        child: AnimatedContainer(
          duration: const Duration(milliseconds: 500),
          curve: Curves.easeInOutQuint,
          margin: EdgeInsets.all(marginValue),
          decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(10),
              color: Colors.black54
          ),

          child: ClipRRect(
            borderRadius: BorderRadius.circular(11),

            child: Stack(
              children: [
                FutureBuilder(
                  future: _initializeVideoPlayerFuture,
                  builder: (context, snapshot) {
                    if (snapshot.connectionState == ConnectionState.done) {
                      return VlcPlayer(
                        controller: _vlcPlayerController,
                        aspectRatio: 16/9,
                        placeholder: const Center(
                          child: CircularProgressIndicator(),
                        ),
                      );
                    } else {
                      return const Center(child: CircularProgressIndicator());
                    }
                  },
                ),

                Positioned(
                  bottom: 10,
                  right: 10,
                  child: InkWell(
                    onTap: () {
                      if(isPlaying) {
                        setState(() {
                          isPlaying = false;
                          myOpacity = 0.75;
                          marginValue = 5.0;
                          buttonColor = secondaryColor;
                        });
                        _vlcPlayerController.pause();
                      } else {
                        setState(() {
                          isPlaying = true;
                          myOpacity = 1.0;
                          marginValue = 0.0;
                          buttonColor = primaryColor;
                        });
                        _vlcPlayerController.play();
                      }
                    },

                    child: Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(100),
                        color: buttonColor
                      ),
                      child: Icon(
                       isPlaying ? Icons.pause : Icons.play_arrow, size: 30, color: bgColor,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}