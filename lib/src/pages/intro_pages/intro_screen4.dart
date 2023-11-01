import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';
import 'package:railway/src/utils/constants.dart';

class IntroPage4 extends StatelessWidget {
  const IntroPage4({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(50),
      decoration: BoxDecoration(
          color: bgColor
      ),
      child: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceAround,

              children: [
                Center(
                  child: Padding(
                    padding: const EdgeInsets.all(15),
                    child: Lottie.asset("assets/animations/detection.json", animate: true, width: 150),
                  ),
                ),

                Padding(
                  padding: const EdgeInsets.only(top: 30, bottom: 15),
                  child: introTextDesign1("Easily detect criminal activities using our strong AL modal"),
                ),

                Center(
                  child: Padding(
                    padding: const EdgeInsets.all(15),
                    child: Lottie.asset("assets/animations/criminal.json", animate: true, width: 150),
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
