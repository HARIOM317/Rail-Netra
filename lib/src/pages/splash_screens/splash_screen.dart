import 'package:animated_splash_screen/animated_splash_screen.dart';
import 'package:flutter/material.dart';
import 'package:railway/src/pages/intro_pages/onboarding_screen.dart';
import 'package:railway/src/utils/constants.dart';

class SplashScreen extends StatelessWidget{
  const SplashScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return AnimatedSplashScreen(
        splash: 'assets/images/icons/logo.png',
        nextScreen: const IntroScreen(),
      splashTransition: SplashTransition.fadeTransition,
      backgroundColor: bgColor,
      splashIconSize: 200,
      duration: 2500,
    );
  }
}