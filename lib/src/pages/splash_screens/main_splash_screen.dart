import 'package:animated_splash_screen/animated_splash_screen.dart';
import 'package:flutter/material.dart';
import 'package:railway/src/pages/drawer_widgets/screen_drawer.dart';
import 'package:railway/src/utils/constants.dart';

class MainSplashScreen extends StatelessWidget{
  const MainSplashScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return AnimatedSplashScreen(
      splash: 'assets/images/icons/logo.png',
      nextScreen: const DrawerScreen(),
      splashTransition: SplashTransition.scaleTransition,
      backgroundColor: bgColor,
      splashIconSize: 200,
      duration: 2500,
    );
  }
}