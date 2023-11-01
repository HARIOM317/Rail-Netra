import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';
import 'package:railway/src/utils/constants.dart';

class IntroPage2 extends StatelessWidget {
  const IntroPage2({super.key});

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
                    child: Lottie.asset("assets/animations/announcement.json", animate: true, width: 225),
                  ),
                ),

                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 15),
                  child: introTextDesign1("We inform the passenger immediately after identifying a less crowded area"),
                ),

                Center(
                  child: Padding(
                    padding: const EdgeInsets.all(15),
                    child: Lottie.asset("assets/animations/crowd.json", animate: true, width: 225),
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
