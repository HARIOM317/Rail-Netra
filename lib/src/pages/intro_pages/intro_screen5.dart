import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';
import 'package:railway/src/utils/constants.dart';

class IntroPage5 extends StatelessWidget {
  const IntroPage5({super.key});

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
                    child: Lottie.asset("assets/animations/security.json", animate: true, width: 225),
                  ),
                ),

                Padding(
                  padding: const EdgeInsets.only(top: 30, bottom: 15),
                  child: introTextDesign1("Let's automate the work with security"),
                )
              ],
            ),
          ),
        ),
      ),
    );
  }
}
