import 'package:flutter/material.dart';
import 'package:railway/src/utils/constants.dart';

class SecondaryButton extends StatelessWidget{
  final String title;
  final Function onPressed;

  const SecondaryButton({super.key, required this.title, required this.onPressed});


  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: FractionalOffset.bottomRight,
      child: SizedBox(
        height: 50,
        child: TextButton(
          onPressed: () {
            onPressed();
          },
          child: Text(title, style: TextStyle(
            color: secondaryColor,
          ),
          ),
        ),
      ),
    );
  }
}