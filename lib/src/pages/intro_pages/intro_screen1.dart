import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:railway/src/utils/constants.dart';

class IntroPage1 extends StatefulWidget {
  const IntroPage1({super.key});

  @override
  State<IntroPage1> createState() => _IntroPage1State();
}

class _IntroPage1State extends State<IntroPage1>
    with SingleTickerProviderStateMixin {
  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(color: bgColor),
      child: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                Center(
                  child: Padding(
                    padding: const EdgeInsets.all(20),
                    child: Image.asset(
                      "assets/images/icons/logo.png",
                      width: 200,
                    ),
                  ),
                ),
                const SizedBox(height: 150),
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Center(
                    child: Text(
                      "Welcome to",
                      style: TextStyle(
                        color: primaryColor.withOpacity(0.8),
                        fontSize: 30,
                        // fontFamily: ''
                      ),
                    ),
                  ),
                ),
                RichText(
                  text: TextSpan(
                      // Default Style
                      style: TextStyle(
                          fontSize: 40,
                          color: primaryColor,
                          fontFamily: 'Roboto-Regular'),
                      children: const <TextSpan>[
                        TextSpan(
                          text: "R",
                          style: TextStyle(
                              fontSize: 50, fontWeight: FontWeight.bold),
                        ),
                        TextSpan(
                          text: "ail ",
                        ),
                        TextSpan(
                          text: "N",
                          style: TextStyle(
                              fontSize: 50, fontWeight: FontWeight.bold),
                        ),
                        TextSpan(
                          text: "etra",
                        )
                      ]),
                ),

                const SizedBox(height: 25,)
              ],
            ),
          ),
        ),
      ),
    );
  }
}
