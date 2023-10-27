import 'package:animated_splash_screen/animated_splash_screen.dart';
import 'package:flutter/material.dart';
import 'package:railway/src/pages/intro_pages/welcome_screen.dart';
import 'package:railway/src/utils/constants.dart';


class LoginWelcomeSplashScreen extends StatelessWidget{
  const LoginWelcomeSplashScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return AnimatedSplashScreen(
      splash: 'assets/images/icons/logo.png',
      nextScreen: const WelcomeScreen(),
      splashTransition: SplashTransition.scaleTransition,
      backgroundColor: bgColor,
      splashIconSize: 200,
      duration: 2500,
    );
  }
}