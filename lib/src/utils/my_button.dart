import 'package:flutter/material.dart';

class MyButton extends StatelessWidget {
  final String iconImagePath;
  final String buttonText;

  const MyButton({super.key, required this.iconImagePath, required this.buttonText});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          height: 75,
          width: 75,
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
              color: const Color(0xfffffafa),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: Colors.black12),
              boxShadow: [
                BoxShadow(
                    color: Colors.grey.shade200,
                    blurRadius: 15,
                    spreadRadius: 3,
                )
              ]
          ),
          child: Center(
            child: Image.asset(iconImagePath),
          ),
        ),

        const SizedBox(height: 10,),

        Text(buttonText, style: TextStyle(
            fontSize: 16,
            color: Colors.grey[700]
        ),
        )
      ],
    );
  }
}
